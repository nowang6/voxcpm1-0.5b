from typing import Tuple

import torch
import torch.nn.functional as F
from torch.func import jvp
from pydantic import BaseModel

from .local_dit import VoxCPMLocDiT


class CfmConfig(BaseModel):
    """条件流匹配（CFM）配置项。

    字段说明:
        sigma_min: 最小噪声尺度（保证数值稳定）
        solver: ODE 求解器类型（如 "euler"）
        t_scheduler: 训练时时间步 t 的采样分布（"log-norm" 或 "uniform"）
        training_cfg_rate: 训练时分类器无关引导（CFG）的 drop 概率（条件置零比例）
        inference_cfg_rate: 推理时 CFG 引导强度
        reg_loss_type: 正则损失类型（如 "l1"）
        ratio_r_neq_t_range: 均值模式下 r!=t 的采样比例区间（随训练进度线性变化）
        noise_cond_prob_range: 对前缀条件加噪的概率区间（随训练进度线性变化）
        noise_cond_scale: 前缀条件加噪的噪声幅度
    """

    sigma_min: float = 1e-6
    solver: str = "euler"
    t_scheduler: str = "log-norm"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 1.0
    reg_loss_type: str = "l1"
    ratio_r_neq_t_range: Tuple[float, float] = (0.25, 0.75)
    noise_cond_prob_range: Tuple[float, float] = (0.0, 0.0)
    noise_cond_scale: float = 0.0


class UnifiedCFM(torch.nn.Module):
    """统一条件流匹配（Unified CFM）模块：包装速度场估计器（LocDiT），实现 ODE 推理求解与训练损失计算。

    推理时从高斯噪声出发，用 Euler 法沿估计的速度场求解常微分方程（ODE），得到流匹配的目标隐变量；
    训练时以条件流匹配的速度回归目标计算损失，可选支持均值模式（jvp 双目标）与 CFG-Zero* 引导。
    """

    def __init__(
        self,
        in_channels: int,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        """初始化统一 CFM。

        参数:
            in_channels: 隐变量通道数 C
            cfm_params: CFM 配置（CfmConfig）
            estimator: 速度场估计器（VoxCPMLocDiT）
            mean_mode: 是否启用均值模式（引入 r、t 两个时刻并用 jvp 计算修正目标）
        """
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.reg_loss_type = cfm_params.reg_loss_type
        self.ratio_r_neq_t_range = cfm_params.ratio_r_neq_t_range
        self.noise_cond_prob_range = cfm_params.noise_cond_prob_range
        self.noise_cond_scale = cfm_params.noise_cond_scale

        self.in_channels = in_channels
        self.mean_mode = mean_mode

        self.estimator = estimator  # 速度场估计器（局部 DiT）

    # ------------------------------------------------------------------ #
    # Inference
    # 推理
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        """CFM 推理入口：从高斯噪声出发，沿估计速度场求解 ODE 生成隐变量。

        参数:
            mu: 全局条件隐变量，形状 (B, C)
            n_timesteps: ODE 积分步数
            patch_size: 待生成分段的帧数 T
            cond: 前缀（上文）条件，形状 (B, C, T')
            temperature: 初始噪声温度（缩放系数）
            cfg_value: 分类器无关引导强度
            sway_sampling_coef: 时间步摆动采样（sway sampling）系数，对时间网格做非均匀重参数化
            use_cfg_zero_star: 是否启用 CFG-Zero* 优化（前期步零初始化 + 无条件速度自适应缩放）

        返回:
            ODE 轨迹的最终状态（生成的隐变量），形状 (B, C, T)
        """
        batch_size, _ = mu.shape

        # 初始噪声状态 (B, C, T)
        noise = torch.randn(
            (batch_size, self.in_channels, patch_size),
            device=mu.device,
            dtype=mu.dtype,
        )
        initial_state = noise * temperature  # 温度控制噪声幅度

        # 时间网格从 1 递减到 0，共 n_timesteps+1 个点
        time_steps = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)

        # sway sampling：余弦重参数化，加密中间时刻的采样密度
        half_pi_t = torch.pi / 2 * time_steps
        sway_offset = torch.cos(half_pi_t) - 1 + time_steps
        time_steps = time_steps + sway_sampling_coef * sway_offset

        return self.solve_euler(
            state=initial_state,
            time_steps=time_steps,
            mu=mu,
            cond=cond,
            cfg_value=cfg_value,
            use_cfg_zero_star=use_cfg_zero_star,
        )

    def optimized_scale(self, conditional_flat: torch.Tensor, unconditional_flat: torch.Tensor):
        """计算 CFG-Zero* 的最优缩放系数：使缩放后的无条件速度最优逼近条件速度（最小二乘）。"""
        dot_product = torch.sum(conditional_flat * unconditional_flat, dim=1, keepdim=True)  # <v_cond, v_uncond> 逐样本内积
        squared_norm = torch.sum(unconditional_flat**2, dim=1, keepdim=True) + 1e-8  # ||v_uncond||^2，+1e-8 防止除零
        zero_star_scale = dot_product / squared_norm  # s = <v_c, v_u> / ||v_u||^2
        return zero_star_scale

    def _estimate_cfg_velocity(
        self,
        state: torch.Tensor,
        current_time: torch.Tensor,
        time_step: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
    ):
        """构造条件/无条件双分支的批量输入，调用估计器并拆分出两个分支的速度。

        Classifier-Free Guidance inference introduced in VoiceBox：前半批为条件分支（保留 mu），
        后半批为无条件分支（mu 置零），状态、时间与前缀条件两分支共享。

        参数:
            state: 当前状态，形状 (B, C, T)
            current_time: 当前时刻 t（标量张量）
            time_step: 当前步长 dt（标量张量）
            mu: 全局条件向量，形状 (B, C)
            cond: 前缀条件，形状 (B, C, T')

        返回:
            (velocity_conditional, velocity_unconditional)：条件速度 v_c 与无条件速度 v_u
        """
        batch_size = state.size(0)

        # 双分支批量输入，各形状在原 batch 维上翻倍 (2B, ...)
        state_input = torch.zeros(
            [2 * batch_size, self.in_channels, state.size(2)],
            device=state.device,
            dtype=state.dtype,
        )  # 状态输入 (2B, C, T)
        condition_input = torch.zeros(
            [2 * batch_size, mu.size(1)],
            device=state.device,
            dtype=state.dtype,
        )
        time_input = torch.zeros([2 * batch_size], device=state.device, dtype=state.dtype)
        time_step_input = torch.zeros([2 * batch_size], device=state.device, dtype=state.dtype)
        audio_condition_input = torch.zeros(
            [2 * batch_size, self.in_channels, cond.size(2)],
            device=state.device,
            dtype=state.dtype,
        )

        # 状态：两个分支共享同一状态
        state_input[:batch_size] = state
        state_input[batch_size:] = state

        # 条件：仅前半批保留条件 mu，后半批为 0（无条件分支）
        condition_input[:batch_size] = mu

        # 前缀条件：两分支均保留（仅 mu 参与引导 drop）
        audio_condition_input[:batch_size] = cond
        audio_condition_input[batch_size:] = cond

        # 时间：两分支共享当前时间 t
        current_time_row = current_time.unsqueeze(0)
        time_input[:batch_size] = current_time_row
        time_input[batch_size:] = current_time_row

        # 步长：两分支共享当前步长 dt
        time_step_row = time_step.unsqueeze(0)
        time_step_input[:batch_size] = time_step_row
        time_step_input[batch_size:] = time_step_row

        # not used now
        if not self.mean_mode:
            time_step_input = torch.zeros_like(time_step_input)  # 非均值模式下不传入步长 dt（当前未启用）

        velocity = self.estimator(state_input, condition_input, time_input, audio_condition_input, time_step_input)

        # 拆分出条件速度 v_c 与无条件速度 v_u
        velocity_conditional, velocity_unconditional = torch.split(velocity, [batch_size, batch_size], dim=0)
        return velocity_conditional, velocity_unconditional

    def _compute_zero_star_scale(self, velocity_conditional: torch.Tensor, velocity_unconditional: torch.Tensor):
        """计算 CFG-Zero* 的缩放系数，并广播回速度张量形状 (B, 1, 1)。"""
        batch_size = velocity_conditional.size(0)

        # (B, C*T) 展平用于计算缩放
        conditional_flat = velocity_conditional.view(batch_size, -1)
        unconditional_flat = velocity_unconditional.view(batch_size, -1)

        # 计算最优缩放系数 s = <v_c, v_u> / ||v_u||^2
        zero_star_scale = self.optimized_scale(conditional_flat, unconditional_flat)

        # 广播回速度张量形状 (B, 1, 1)
        broadcast_dims = [1] * (velocity_conditional.dim() - 1)
        zero_star_scale = zero_star_scale.view(batch_size, *broadcast_dims)
        return zero_star_scale

    def solve_euler(
        self,
        state: torch.Tensor,
        time_steps: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        """Euler 法求解流匹配 ODE：dx/dt = v(x, t)，从 t=1 积分到 t=0。

        参数:
            state: 初始噪声状态，形状 (B, C, T)
            time_steps: 单调递减的时间网格（经 sway sampling 重参数化）
            mu: 全局条件向量，形状 (B, C)
            cond: 前缀条件，形状 (B, C, T')
            cfg_value: 分类器无关引导强度
            use_cfg_zero_star: 是否启用 CFG-Zero*（前几步速度置零 + 无条件速度自适应缩放）

        返回:
            ODE 轨迹的最终状态（生成的隐变量），形状 (B, C, T)
        """
        # 当前时刻 t 与初始步长 dt（网格可能非均匀，步长随后按网格更新）
        current_time = time_steps[0]
        time_step = time_steps[0] - time_steps[1]

        trajectory = []
        zero_init_steps = max(1, int(len(time_steps) * 0.04))  # CFG-Zero*：前 4%（magic number 0.04）的步数将速度置零

        for step_index in range(1, len(time_steps)):
            use_zero_init = use_cfg_zero_star and step_index <= zero_init_steps

            if use_zero_init:
                # 初期速度置零，避免噪声初期误差累积（CFG-Zero* 的 zero-init）
                velocity_combined = torch.zeros_like(state)
            else:
                # 分别得到条件速度 v_c 与无条件速度 v_u
                velocity_conditional, velocity_unconditional = self._estimate_cfg_velocity(
                    state, current_time, time_step, mu, cond
                )

                if use_cfg_zero_star:
                    zero_star_scale = self._compute_zero_star_scale(velocity_conditional, velocity_unconditional)
                else:
                    zero_star_scale = 1.0  # 不启用时缩放为 1，退化为普通 CFG

                # CFG-Zero* 组合：v = s*v_u + cfg*(v_c - s*v_u)
                scaled_unconditional = velocity_unconditional * zero_star_scale
                guidance_residual = velocity_conditional - scaled_unconditional
                velocity_combined = scaled_unconditional + cfg_value * guidance_residual

            # Euler 单步更新：x_{t-dt} = x_t - dt * v(x_t, t)
            state = state - time_step * velocity_combined

            # 时间推进
            current_time = current_time - time_step

            # 记录轨迹
            trajectory.append(state)

            # 按时间网格更新下一步长 dt（网格可能非均匀）
            if step_index < len(time_steps) - 1:
                time_step = current_time - time_steps[step_index + 1]

        return trajectory[-1]

    # ------------------------------------------------------------------ #
    # Training loss
    # 训练损失
    # ------------------------------------------------------------------ #
    def adaptive_loss_weighting(
        self, losses: torch.Tensor, mask: torch.Tensor | None = None, p: float = 0.0, epsilon: float = 1e-3
    ):
        """自适应损失加权：损失越小权重越大（p=0 时退化为均匀权重）。

        参数:
            losses: 逐样本损失，形状 (B, T)
            mask: 有效帧掩码，None 表示全部有效
            p: 加权指数，weights = 1/(loss+eps)^p
            epsilon: 防止除零的小量

        返回:
            与 losses 同形的权重张量（已 detach，不参与梯度）
        """
        weights = 1.0 / ((losses + epsilon).pow(p))
        if mask is not None:
            weights = weights * mask  # 无效帧权重置零
        return weights.detach()

    def sample_r_t(self, x: torch.Tensor, mu: float = -0.4, sigma: float = 1.0, ratio_r_neq_t: float = 0.0):
        """采样流匹配的两个时刻 r 与 t。

        参数:
            x: 输入张量（仅用于获取批大小与设备信息）
            mu: log-norm 采样分布的均值（magic number，默认 -0.4 使 t 经 sigmoid 后偏向小值区间）
            sigma: log-norm 采样分布的标准差
            ratio_r_neq_t: r!=t 的采样比例，其余样本强制 r=t（退化为单时刻 CFM）

        返回:
            r, t: 形状 (B,) 的两个时刻张量（r<=t）
        """
        batch_size = x.shape[0]
        if self.t_scheduler == "log-norm":
            s_r = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu  # sigmoid 前的 logits 正态采样
            s_t = torch.randn(batch_size, device=x.device, dtype=x.dtype) * sigma + mu
            r = torch.sigmoid(s_r)
            t = torch.sigmoid(s_t)  # sigmoid 压缩到 (0, 1)，分布偏向小值
        elif self.t_scheduler == "uniform":
            r = torch.rand(batch_size, device=x.device, dtype=x.dtype)  # [0, 1) 均匀采样
            t = torch.rand(batch_size, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported t_scheduler: {self.t_scheduler}")

        mask = torch.rand(batch_size, device=x.device, dtype=x.dtype) < ratio_r_neq_t  # 按比例决定该样本是否 r!=t
        r, t = torch.where(
            mask,
            torch.stack([torch.min(r, t), torch.max(r, t)], dim=0),  # r!=t 时取 (min, max) 保证 r<=t
            torch.stack([t, t], dim=0),  # 否则强制 r=t
        )

        return r.squeeze(), t.squeeze()

    def compute_loss(
        self,
        x1: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
        progress: float = 0.0,
    ):
        """计算条件流匹配（CFM）训练损失。

        参数:
            x1: 目标干净隐变量，形状 (B, C, T)
            mu: 全局条件向量，形状 (B, C)
            cond: 前缀条件，形状 (B, C, T')，可为 None
            tgt_mask: 目标有效帧掩码，形状 (B, 1, T)，可为 None
            progress: 训练进度 [0, 1]，用于调度 r!=t 比例与前缀条件加噪概率

        返回:
            标量损失（速度回归 MSE，可选自适应加权与掩码归一化）
        """
        batch_size, _, _ = x1.shape

        if self.training_cfg_rate > 0:
            cfg_mask = torch.rand(batch_size, device=x1.device) > self.training_cfg_rate  # True 表示保留条件
            mu = mu * cfg_mask.view(-1, 1)  # 分类器无关引导训练：按概率将部分样本的条件置零，学习无条件分支

        if cond is None:
            cond = torch.zeros_like(x1)  # 无前缀条件时用零张量占位

        noisy_mask = torch.rand(batch_size, device=x1.device) > (
            1.0
            - (
                self.noise_cond_prob_range[0]
                + progress * (self.noise_cond_prob_range[1] - self.noise_cond_prob_range[0])
            )
        )  # 以随训练进度线性变化的概率选中需加噪的样本
        cond = cond + noisy_mask.view(-1, 1, 1) * torch.randn_like(cond) * self.noise_cond_scale  # 条件加噪增强（noise augmentation）

        ratio_r_neq_t = (
            self.ratio_r_neq_t_range[0] + progress * (self.ratio_r_neq_t_range[1] - self.ratio_r_neq_t_range[0])
            if self.mean_mode
            else 0.0
        )  # 均值模式下按训练进度调度 r!=t 的比例

        r, t = self.sample_r_t(x1, ratio_r_neq_t=ratio_r_neq_t)
        r_ = r.detach().clone()
        t_ = t.detach().clone()  # 备份不参与梯度的时间张量
        z = torch.randn_like(x1)  # 标准高斯噪声 x0
        y = (1 - t_.view(-1, 1, 1)) * x1 + t_.view(-1, 1, 1) * z  # 线性插值构造流上的中间点：x_t = (1-t)*x1 + t*x0
        v = z - x1  # 目标速度场：v* = x0 - x1（线性流的真实瞬时速度）

        def model_fn(z_sample, r_sample, t_sample):
            """调用速度场估计器，预测 (y, r, t) 处的速度。"""
            return self.estimator(z_sample, mu, t_sample, cond, time_step=t_sample - r_sample)

        if self.mean_mode:
            v_r = torch.zeros_like(r)  # r 方向扰动为 0
            v_t = torch.ones_like(t)  # t 方向扰动为 1：沿 t 求方向导数
            from torch.backends.cuda import sdp_kernel

            with sdp_kernel(enable_flash=False, enable_mem_efficient=False):  # 关闭 flash/高效注意力后端以保证 jvp 兼容
                u_pred, dudt = jvp(model_fn, (y, r, t), (v, v_r, v_t))  # jvp：同时得到速度预测 u 及其对 t 的方向导数 du/dt
            u_tgt = v - (t_ - r_).view(-1, 1, 1) * dudt  # 均值模式目标：u* = v - (t-r)*du/dt（对区间 [r, t] 的平均速度修正）
        else:
            u_pred = model_fn(y, r, t)
            u_tgt = v  # 普通模式：直接回归瞬时速度 v

        losses = F.mse_loss(u_pred, u_tgt.detach(), reduction="none").mean(dim=1)  # 逐样本 MSE，通道维平均 -> (B, T)
        if tgt_mask is not None:
            weights = self.adaptive_loss_weighting(losses, tgt_mask.squeeze(1))
            loss = (weights * losses).sum() / torch.clamp(torch.sum(tgt_mask), min=1.0)  # 仅对有效帧加权求平均（分母为有效帧数）
        else:
            loss = losses.mean()  # 无掩码时直接取均值

        return loss
