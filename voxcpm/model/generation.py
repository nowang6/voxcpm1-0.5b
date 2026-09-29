"""
VoxCPM1 (0.5B) 的生成与推理逻辑（VoxCPMGenerationMixin）。

包含：提示缓存构建/合并、输入序列拼装、坏例重试、以及逐 patch 的
自回归推理循环。宿主类为 voxcpm.model.voxcpm.VoxCPMModel。

从 OpenBMB/VoxCPM 抽取而来，仅保留单一架构（voxcpm v1 / VoxCPM-0.5B）。
"""

import sys
import warnings
from typing import Generator, List, Optional, Tuple, Union

import torch
from einops import rearrange
from tqdm import tqdm

from .utils import (
    apply_generation_seed,
    consume_single_result,
    get_dtype,
    load_audio,
    materialize_generation_seed,
)


class VoxCPMGenerationMixin:
    """VoxCPMModel 的生成/推理混入类（Mixin）。

    宿主类需提供以下属性：
    - 配置与运行时：config、device、patch_size、chunk_size、sample_rate
    - 分词：text_tokenizer、audio_start_token、last_successful_seed
    - 子模块：base_lm、residual_lm、feat_encoder、feat_decoder、audio_vae、
      fsq_layer、enc_to_lm_proj、lm_to_dit_proj、res_to_dit_proj、
      stop_proj、stop_actn、stop_head
    """

    # ------------------------------------------------------------------
    # 公开生成入口
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs) -> torch.Tensor:
        """一次性（非流式）生成完整音频张量。"""
        return consume_single_result(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs) -> Generator[torch.Tensor, None, None]:
        """流式生成音频，逐块 yield 已解码的波形张量。"""
        return self._generate(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,  # setting acceptable ratio of audio length to text length (for badcase detection)
        streaming: bool = False,
        seed: Optional[int] = None,
    ) -> Generator[torch.Tensor, None, None]:
        """生成入口（生成器）：支持零样本与声音克隆（提示音频）两种输入。

        内部先把提示音频编码为特征、组装成提示缓存，再复用
        ``_generate_with_prompt_cache`` 的统一生成流程。

        Args:
            target_text: 待合成的目标文本。
            prompt_text: 提示（音频）对应的文本（声音克隆时提供）。
            prompt_wav_path: 提示（音频）路径；为空表示零样本（不带参考音色）生成。
            min_len: 最小生成长度（token 数），未达到前不判停止。
            max_len: 最大生成长度上限。
            inference_timesteps: 流匹配 ODE 求解步数。
            cfg_value: 无分类器引导（CFG）强度。
            retry_badcase: 是否启用坏例重试。
            retry_badcase_max_times: 坏例重试的最大次数。
            retry_badcase_ratio_threshold: 音频/文本长度比的阈值（坏例检测用）。
            streaming: 是否流式逐块输出。
            seed: 随机种子（None 时随机生成）。
        Yields:
            torch.Tensor: 一维波形张量（CPU）。
        """
        if prompt_wav_path:
            # 声音克隆：把提示音频编码为 patch 特征，包装成提示缓存
            prompt_cache = {
                "prompt_text": prompt_text,
                "audio_feat": self._encode_prompt_audio_feat(prompt_wav_path),
            }
        else:
            # 零样本生成：无提示缓存
            prompt_cache = None
        gen = self._generate_with_prompt_cache(
            target_text=target_text,
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            streaming=streaming,
            seed=seed,
            # 保持 _generate 的既有行为：非流式输出不裁掉开头复述提示上下文的音频段
            trim_prompt_prefix=False,
        )
        for decode_audio, _, _ in gen:
            yield decode_audio

    # ------------------------------------------------------------------
    # 提示缓存（声音克隆）
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def build_prompt_cache(
        self,
        prompt_text: str,
        prompt_wav_path: str,
    ):
        """
        构建提示（音频）缓存，供后续快速生成复用。

        Args:
            prompt_text: 提示文本（必填）。
            prompt_wav_path: 提示音频路径（必填）。

        Returns:
            dict: 提示缓存，包含 prompt_text（原始文本）与 audio_feat（音频特征）。
                 文本分词延迟到生成阶段进行，以保证分词一致性。
        """
        if not prompt_text or not prompt_wav_path:
            raise ValueError("prompt_text and prompt_wav_path are required")
        # 构建提示缓存：仅保存原始文本与音频特征（避免缓存与生成时分词不一致）
        return {
            "prompt_text": prompt_text,
            "audio_feat": self._encode_prompt_audio_feat(prompt_wav_path),
        }

    def merge_prompt_cache(
        self,
        original_cache: dict,
        new_text: str,
        new_audio_feat: torch.Tensor,
    ):
        """将新生成的内容合并进原提示缓存，用于稳定音色（滚动上下文）。"""
        if original_cache is None:
            # 无原缓存时，直接用新内容初始化
            return {
                "prompt_text": new_text,
                "audio_feat": new_audio_feat,
            }
        original_prompt_text = original_cache["prompt_text"]
        original_audio_feat = original_cache["audio_feat"]
        # Merge text by concatenation
        # 文本按字符串拼接，音频特征沿时间维拼接
        merged_prompt_text = original_prompt_text + new_text
        merged_audio_feat = torch.cat([original_audio_feat, new_audio_feat], dim=0)

        # build new cache
        # 构建新缓存
        merged_cache = {
            "prompt_text": merged_prompt_text,
            "audio_feat": merged_audio_feat,
        }

        return merged_cache

    def generate_with_prompt_cache(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用预构建的提示缓存一次性生成（非流式）。"""
        return consume_single_result(self._generate_with_prompt_cache(*args, streaming=False, **kwargs))

    def generate_with_prompt_cache_streaming(
        self, *args, **kwargs
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], None, None]:
        """使用预构建的提示缓存流式生成。"""
        return self._generate_with_prompt_cache(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _generate_with_prompt_cache(
        self,
        target_text: str,
        prompt_cache: dict,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        retry_badcase: bool = False,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
        seed: Optional[int] = None,
        *,
        trim_prompt_prefix: bool = True,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """
        使用预构建的提示缓存生成音频（生成器），是所有生成路径的统一实现。

        与 _generate 的区别：提示（音频）的编码结果直接来自缓存，
        省去每次生成都重新加载/编码参考音频的开销。

        Args:
            trim_prompt_prefix: 非流式输出是否裁掉开头 (streaming_prefix_len - 1) 个
                chunk 的提示复述段。默认 True（提示缓存路径的既有行为）；
                _generate 出于历史行为保持 False。

        Yields:
            Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
                (波形张量, 目标文本 token, 生成的音频特征序列)。
        """
        if retry_badcase and streaming:
            # 流式模式下无法回滚已输出的音频，因此不支持坏例重试
            warnings.warn("Retry on bad cases is not supported in streaming mode, setting retry_badcase=False.")
            retry_badcase = False
        # get prompt from cache
        # 从缓存取出提示信息；无缓存则退化为零样本生成
        if prompt_cache is None:
            prompt_audio_feat = torch.empty((0, self.patch_size, self.audio_vae.latent_dim), dtype=torch.float32)
            text = target_text
        else:
            prompt_audio_feat = prompt_cache["audio_feat"]
            prompt_text = prompt_cache["prompt_text"]
            text = prompt_text + target_text

        target_text_token = torch.LongTensor(self.text_tokenizer(target_text))

        # 按零填充的方式对齐文本段与音频段（音频位置的 token 用 0 占位，
        # 文本段的音频特征用全零占位），掩码标记各自的有效位置
        text_token, text_mask, audio_feat, audio_mask = self._assemble_sequence_inputs(text, prompt_audio_feat)

        # run inference
        # ---------- 坏例重试循环：用不同随机种子反复生成，直到长度比可接受 ----------
        target_text_length = len(self.text_tokenizer(target_text))
        retry_badcase_times = 0
        current_seed = materialize_generation_seed(seed)
        last_attempt_seed = current_seed
        while retry_badcase_times < retry_badcase_max_times:
            last_attempt_seed = current_seed
            apply_generation_seed(last_attempt_seed)  # 写入随机状态保证可复现

            inference_result = self._inference(
                text_token,
                text_mask,
                audio_feat,
                audio_mask,
                min_len=min_len,
                # 动态长度上限：文本 token 数 × 阈值 + 10，避免异常超长音频
                max_len=min(
                    int(target_text_length * retry_badcase_ratio_threshold + 10), max_len
                ),  # avoid too long audio
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if streaming:
                # 流式模式：每个新 patch 的隐变量立即解码成波形并 yield
                patch_len = self.patch_size * self.chunk_size
                for latent_pred, pred_audio_feat in inference_result:
                    decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
                    decode_audio = decode_audio[..., -patch_len:].squeeze(1).cpu()  # 只保留最新 chunk 的波形
                    self.last_successful_seed = last_attempt_seed
                    yield (decode_audio, target_text_token, pred_audio_feat)
                break
            else:
                latent_pred, pred_audio_feat = consume_single_result(inference_result)
                if retry_badcase:
                    # 坏例判定：音频 token 数超过 文本 token 数 × 阈值则换种子重试
                    if pred_audio_feat.shape[0] >= target_text_length * retry_badcase_ratio_threshold:
                        print(
                            f"  Badcase detected, audio_text_ratio={pred_audio_feat.shape[0] / target_text_length}, retrying...",
                            file=sys.stderr,
                        )
                        retry_badcase_times += 1
                        current_seed += 1
                        continue
                    else:
                        break
                else:
                    break
        if not streaming:
            # 非流式：整段隐变量一次性解码为波形
            self.last_successful_seed = last_attempt_seed
            decode_audio = self.audio_vae.decode(latent_pred.to(torch.float32))
            patch_len = self.patch_size * self.chunk_size
            if trim_prompt_prefix and audio_mask.sum().item() > 0:
                # 有提示音频时：流式解码从第 streaming_prefix_len 个 chunk 才产出真正的目标音频，
                # 前 (streaming_prefix_len - 1) 个 chunk 属于被当作上下文复述的提示内容，需裁掉
                decode_audio = decode_audio[..., patch_len * (streaming_prefix_len - 1) :].squeeze(1).cpu()
            else:
                decode_audio = decode_audio[..., :].squeeze(1).cpu()
            yield (decode_audio, target_text_token, pred_audio_feat)

    # ------------------------------------------------------------------
    # 输入拼装
    # ------------------------------------------------------------------

    def _encode_prompt_audio_feat(self, prompt_wav_path: str) -> torch.Tensor:
        """加载提示音频并编码为 patch 化的隐变量特征。

        Args:
            prompt_wav_path: 提示音频路径。

        Returns:
            torch.Tensor: 形状 (T, patch_size, latent_dim) 的特征（CPU）。
        """
        # load audio
        # 加载提示音频并重采样到模型采样率
        audio, sr = load_audio(prompt_wav_path, target_sr=self.sample_rate)

        patch_len = self.patch_size * self.chunk_size

        if audio.size(1) % patch_len != 0:
            # Left padding: pad at the beginning of the audio to keep valid audio data at the end of the sequence
            # 左填充：在音频开头填充，保持有效音频数据在序列末尾
            padding_size = patch_len - audio.size(1) % patch_len
            audio = torch.nn.functional.pad(audio, (padding_size, 0))

        # extract audio features
        # 提取音频隐变量特征并按 patch 组织
        audio_feat = self.audio_vae.encode(audio.to(self.device), self.sample_rate).cpu()
        audio_feat = audio_feat.view(
            self.audio_vae.latent_dim,
            -1,
            self.patch_size,
        ).permute(1, 2, 0)  # (T, P, D)
        return audio_feat

    def _assemble_sequence_inputs(
        self,
        text: str,
        prompt_audio_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """把「文本 + 提示音频特征」拼装为模型输入张量。

        文本分词后在末尾拼接音频起始 token；音频位置用 0 占位 token 与
        全零特征对齐；掩码标记文本段/音频段各自的有效位置。

        Args:
            text: 完整文本（提示文本 + 目标文本）。
            prompt_audio_feat: 提示音频特征 (T_audio, patch_size, latent_dim)；
                零样本时传长度为 0 的空张量。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (text_token (1, T), text_mask (1, T), audio_feat (1, T, P, D),
                 audio_mask (1, T))，均已搬到运行设备。
        """
        # 「文本」整体分词，末尾拼上音频起始 token
        text_token = torch.LongTensor(self.text_tokenizer(text))
        text_token = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_start_token],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )

        # 按零填充的方式对齐文本段与音频段
        audio_length = prompt_audio_feat.size(0)
        text_length = text_token.shape[0]
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        audio_pad_feat = torch.zeros(
            (text_length, self.patch_size, self.audio_vae.latent_dim),
            dtype=torch.float32,
            device=text_token.device,
        )
        text_token = torch.cat([text_token, text_pad_token])
        audio_feat = torch.cat([audio_pad_feat, prompt_audio_feat], dim=0)
        # 掩码构造：前 text_length 个位置为文本，后 audio_length 个位置为音频
        text_mask = (
            torch.cat([torch.ones(text_length), torch.zeros(audio_length)]).type(torch.int32).to(text_token.device)
        )
        audio_mask = (
            torch.cat([torch.zeros(text_length), torch.ones(audio_length)]).type(torch.int32).to(text_token.device)
        )

        # 增加 batch 维并搬到运行设备
        return (
            text_token.unsqueeze(0).to(self.device),
            text_mask.unsqueeze(0).to(self.device),
            audio_feat.unsqueeze(0).to(self.device).to(get_dtype(self.config.dtype)),
            audio_mask.unsqueeze(0).to(self.device),
        )

    # ------------------------------------------------------------------
    # 核心推理
    # ------------------------------------------------------------------

    def inference(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """非流式核心推理：返回完整音频特征张量。"""
        return consume_single_result(self._inference(*args, streaming=False, **kwargs))

    def inference_streaming(self, *args, **kwargs) -> Generator[Tuple[torch.Tensor, List[torch.Tensor]], None, None]:
        """流式核心推理：逐 patch yield 音频特征。"""
        return self._inference(*args, streaming=True, **kwargs)

    @torch.inference_mode()
    def _inference(
        self,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        feat: torch.Tensor,
        feat_mask: torch.Tensor,
        min_len: int = 2,
        max_len: int = 2000,
        inference_timesteps: int = 10,
        cfg_value: float = 2.0,
        streaming: bool = False,
        streaming_prefix_len: int = 3,
    ) -> Generator[Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]], None, None]:
        """核心推理循环：LM 逐 patch 自回归预测，LocDiT 用流匹配渲染每个 patch。

        整体流程：
        1. prefill：整段（文本+提示音频）一次前向，结果写入两个 LM 的 KV 缓存；
        2. 逐 patch 生成：LM 隐状态 -> DiT 条件 -> 流匹配 ODE 采样出 patch 特征
           -> 特征重新编码 -> 单步前向更新 KV 缓存，循环直至停止头触发或达到 max_len。

        Args:
            text: token 序列，形状 (1, T)。
            text_mask: 文本位置掩码，形状 (1, T)。
            feat: 音频 patch 特征，形状 (1, T, P, D)。
            feat_mask: 音频位置掩码，形状 (1, T)。
            min_len: 最小生成长度（此前不判停止，避免过早截断）。
            max_len: 最大生成 patch 数。
            inference_timesteps: 流匹配 ODE 求解步数。
            cfg_value: 无分类器引导（CFG）强度。
            streaming: 是否流式逐块输出。
            streaming_prefix_len: 流式上下文 patch 数（含当前 patch，见循环内注释）。
        Yields:
            Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
                流式时为（最近 chunk 特征 (1, D, t*P)，patch 列表）；
                非流式时为（完整特征 (1, D, T*P)，patch 序列张量 (T, P, D)）。
        """
        B, T, P, D = feat.shape  # (batch=1, 序列长度, patch 大小, 特征维度)

        # prefill 用未编译的原始编码器（形状多变时比编译版更稳），生成循环内再用编译版
        prefill_encoder = getattr(self, "_feat_encoder_raw", self.feat_encoder)
        feat_embed = prefill_encoder(feat)  # [b, t, h_feat]
        feat_embed = self.enc_to_lm_proj(feat_embed)  # 投影到 LM 隐藏维度

        # muP 场景按 scale_emb 缩放词嵌入，否则不缩放
        if self.config.lm_config.use_mup:
            scale_emb = self.config.lm_config.scale_emb
        else:
            scale_emb = 1.0

        text_embed = self.base_lm.embed_tokens(text) * scale_emb
        # 按 mask 融合：文本位置取词嵌入，音频位置取音频特征嵌入
        combined_embed = text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed

        prefix_feat_cond = feat[:, -1, ...]  # b, p, d：首个 patch 的流匹配条件（序列最后一个已知 patch）
        pred_feat_seq = []  # b, t, p, d：已生成 patch 的列表
        curr_embed = None

        # Prepare prompt context patches for streaming mode
        # When there's a prompt audio, use its last (streaming_prefix_len - 1) patches as initial context
        # 为流式模式准备提示上下文 patch：
        # 有提示音频时，取其最后 (streaming_prefix_len - 1) 个 patch 作为初始上下文
        prompt_context_patches = []
        audio_patch_count = int(feat_mask.sum().item())  # 提示音频的 patch 数
        if audio_patch_count > 0:
            context_len = min(streaming_prefix_len - 1, audio_patch_count)
            # Take the last context_len patches from prompt audio as initial context
            # Split into list of [b, 1, p, d] tensors to match pred_feat_seq format
            # 取提示音频最后 context_len 个 patch 作为初始上下文，
            # 并拆成 [b, 1, p, d] 的列表以对齐 pred_feat_seq 的格式
            prompt_context_patches = list(feat[:, -context_len:, :, :].split(1, dim=1))
            pred_feat_seq = prompt_context_patches + pred_feat_seq

        # ---------- prefill：整段前向并填充基础 LM 的 KV 缓存 ----------
        enc_outputs, kv_cache_tuple = self.base_lm(
            inputs_embeds=combined_embed,
            is_causal=True,
        )
        self.base_lm.kv_cache.fill_caches(kv_cache_tuple)

        # 音频位置走 FSQ 量化，文本位置保持原隐状态
        enc_outputs = self.fsq_layer(enc_outputs) * feat_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]  # 取最后一个位置的隐状态作为首个生成步的条件

        # ---------- prefill：残差声学 LM 同样填充 KV 缓存 ----------
        residual_enc_outputs, residual_kv_cache_tuple = self.residual_lm(
            inputs_embeds=enc_outputs + feat_mask.unsqueeze(-1) * feat_embed,
            is_causal=True,
        )
        self.residual_lm.kv_cache.fill_caches(residual_kv_cache_tuple)
        residual_hidden = residual_enc_outputs[:, -1, :]

        # ---------- 逐 patch 自回归生成循环 ----------
        for i in tqdm(range(max_len)):
            dit_hidden_1 = self.lm_to_dit_proj(lm_hidden)  # [b, h_dit]
            dit_hidden_2 = self.res_to_dit_proj(residual_hidden)  # [b, h_dit]
            dit_hidden = dit_hidden_1 + dit_hidden_2  # [b, h_dit] 双路条件相加

            # 流匹配 ODE 采样：以 dit_hidden 为均值条件、上一个 patch 为条件特征，
            # 经 inference_timesteps 步 ODE 求解生成当前 patch（cfg_value 控制 CFG 引导强度）
            pred_feat = self.feat_decoder(
                mu=dit_hidden,
                patch_size=self.patch_size,
                cond=prefix_feat_cond.transpose(1, 2).contiguous(),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(
                1, 2
            )  # [b, p, d]

            # 生成的 patch 重新编码为嵌入，供下一步 LM 前向使用（自回归闭环）
            curr_embed = self.feat_encoder(pred_feat.unsqueeze(1))  # b, 1, c
            curr_embed = self.enc_to_lm_proj(curr_embed)

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # b, 1, p, d
            prefix_feat_cond = pred_feat  # 更新流匹配条件为最新 patch

            if streaming:
                # return the last three predicted latent features to provide enough context for smooth decoding
                # 取最近 streaming_prefix_len 个 patch 交给 VAE 解码：
                # VAE 解码需要前后上下文，多给几个 patch 才能保证拼接处平滑无缝
                pred_feat_chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)
                feat_pred = rearrange(pred_feat_chunk, "b t p d -> b d (t p)", b=B, p=self.patch_size)

                yield feat_pred, pred_feat_seq

            # 停止预测头判断是否结束（输出 2 类 logits，取 argmax）
            stop_flag = self.stop_head(self.stop_actn(self.stop_proj(lm_hidden))).argmax(dim=-1)[0].cpu().item()
            if i > min_len and stop_flag == 1:
                break  # 超过最小长度且预测为"停止"，结束生成

            # 基础 LM 单步前向：复用 prefill 填充的 KV 缓存，只计算新 token 的增量注意力
            lm_hidden = self.base_lm.forward_step(
                curr_embed[:, 0, :], torch.tensor([self.base_lm.kv_cache.step()], device=curr_embed.device)
            ).clone()

            lm_hidden = self.fsq_layer(lm_hidden)  # 新隐状态同样过 FSQ 量化
            # 残差声学 LM 单步前向：以量化隐状态 + 当前 patch 嵌入为输入，复用其 KV 缓存
            residual_hidden = self.residual_lm.forward_step(
                lm_hidden + curr_embed[:, 0, :],
                torch.tensor([self.residual_lm.kv_cache.step()], device=curr_embed.device),
            ).clone()

        if not streaming:
            # 非流式：把所有 patch 沿时间维拼接成完整序列
            pred_feat_seq = torch.cat(pred_feat_seq, dim=1)  # b, t, p, d
            feat_pred = rearrange(pred_feat_seq, "b t p d -> b d (t p)", b=B, p=self.patch_size)
            yield feat_pred, pred_feat_seq.squeeze(0).cpu()
