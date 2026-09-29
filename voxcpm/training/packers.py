"""序列打包（packing）实现：把文本 token 与音频波形组合、对齐、生成各类 mask，
输出 ``VoxCPMModel.forward()`` 直接可用的批张量。"""
from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange


class AudioFeatureProcessingPacker:
    """
    把原始文本 token id 与音频波形转换成 ``VoxCPMModel.forward()``
    所需的填充后的张量。

    每个样本的序列布局（sequence layout）：
        [文本 token, 101, 音频特征, 102]

    损失（diffusion + stop）只在音频段上计算。
    """

    def __init__(self, max_len: int, patch_size: int, feat_dim: int, audio_vae: nn.Module):
        """
        Args:
            max_len: 打包后序列的最大长度（超出部分截断）。
            patch_size: 每个序列位置对应的 VAE 潜变量帧数（patch 大小）。
            feat_dim: VAE 潜变量的特征维度。
            audio_vae: 用于把波形编码为潜变量特征的 AudioVAE 模块。
        """
        # 音频段起始/结束的特殊 token id
        self.audio_start_id = 101
        self.audio_end_id = 102

        self.patch_size = patch_size
        # 一个序列位置覆盖的原始波形采样点数 = VAE hop_length × patch_size
        self.patch_len = audio_vae.hop_length * self.patch_size
        self.feat_dim = feat_dim
        self.max_len = max_len

        self.audio_vae = audio_vae

    # ------------------------------------------------------------------ #
    # Helpers
    # 辅助方法
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_pad_position(tokens: torch.Tensor):
        """返回序列中第一个填充位置（-100）的下标；没有填充时返回 None。"""
        # 找出所有 -100（填充值）的下标，最靠前的一个即为有效序列长度
        positions = (tokens == -100).nonzero(as_tuple=True)
        if positions[0].numel() == 0:
            return None
        return int(positions[0][0])

    def unpad_text_tokens(self, tokens: torch.Tensor):
        """去掉文本 token 序列末尾的填充（-100），仅保留有效部分。"""
        pad_pos = self._first_pad_position(tokens)
        return tokens if pad_pos is None else tokens[:pad_pos]

    def unpad_audio_tokens(self, tokens: torch.Tensor):
        """去掉音频 token 序列末尾的填充（-100），仅保留有效部分。"""
        pad_pos = self._first_pad_position(tokens)
        return tokens if pad_pos is None else tokens[:pad_pos]

    def encode_audio(self, wav: torch.Tensor):
        """
        使用 AudioVAE 把原始波形编码为潜变量特征。

        AudioVAE.encode 期望输入形状 [B, 1, T']，返回 [B, D, T]；
        随后转置为 [B, T, D] 以符合下游的形状约定。
        """
        wav = wav.unsqueeze(0)  # [1, T]
        wav = wav.unsqueeze(1)  # [1, 1, T]
        wav_len = wav.size(-1)
        # 波形采样点数不足一个序列位置覆盖的 patch_len 时，末尾补零对齐
        if wav_len % self.patch_len != 0:
            padding_size = self.patch_len - wav_len % self.patch_len
            wav = torch.nn.functional.pad(wav, (0, padding_size))

        with torch.no_grad():
            # VAE 编码仅用于构造训练目标，不参与梯度计算
            z = self.audio_vae.encode(wav, self.audio_vae.sample_rate)  # [1, D, T']
            feat = z.transpose(1, 2)  # [1, T', D]
        return feat

    # ------------------------------------------------------------------ #
    # Main entry point
    # 主入口
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        基于填充（padding）的批处理：批内每个样本先独立完成序列打包，
        再统一填充到公共长度（以 ``max_len`` 为上限）。

        Args:
            audio_tokens: [B, T] 音频波形批（含 -100 填充）。
            text_tokens: [B, T] 文本 token id 批（含 -100 填充）。

        Returns:
            打包后的批张量字典：text_tokens / audio_feats / text_mask /
            audio_mask / loss_mask / position_ids / labels。
        """
        device = audio_tokens.device

        # 逐样本保存打包结果：拼接后的 token、音频特征、三种 mask、labels 与序列长度
        text_tokens_list: List[torch.Tensor] = []
        audio_feats_list: List[torch.Tensor] = []
        text_mask_list: List[torch.Tensor] = []
        audio_mask_list: List[torch.Tensor] = []
        loss_mask_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        lengths: List[int] = []

        # 逐样本处理：先去掉 -100 填充，再拼接序列、提取音频特征并生成 mask
        for audio_token, text_token in zip(audio_tokens, text_tokens):
            (
                packed_text,
                audio_feat,
                text_mask,
                audio_mask,
                loss_mask,
                labels,
            ) = self.process_tts_data(
                # 音频转为 float32 波形，文本保持 token id
                self.unpad_audio_tokens(audio_token).to(torch.float32),
                self.unpad_text_tokens(text_token),
            )

            text_tokens_list.append(packed_text)
            text_mask_list.append(text_mask)
            audio_feats_list.append(audio_feat)
            audio_mask_list.append(audio_mask)
            loss_mask_list.append(loss_mask)
            labels_list.append(labels)
            # 记录打包后的总序列长度（文本 + 101 + 音频 + 102）
            lengths.append(packed_text.shape[0])

        # Determine padded length per batch (cap by self.max_len)
        # 本批统一的填充长度：取批内最大长度，且不超过 self.max_len
        if lengths:
            max_len = min(self.max_len, max(lengths))
        else:
            max_len = self.max_len

        def pad_1d(x: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
            """把一维张量截断/填充到 max_len，填充值为 pad_value。"""
            if x.size(0) >= max_len:
                return x[:max_len]
            pad = torch.full((max_len - x.size(0),), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)

        def pad_3d(x: torch.Tensor) -> torch.Tensor:
            """把 [T, P, D] 的音频特征张量截断/零填充到 max_len。"""
            # x: [T, P, D]
            if x.size(0) >= max_len:
                return x[:max_len]
            pad = torch.zeros((max_len - x.size(0),) + x.shape[1:], dtype=x.dtype, device=x.device)
            return torch.cat([x, pad], dim=0)

        if lengths:
            # 逐样本填充后堆叠成批张量：token/mask/labels 为 [B, max_len]，音频特征为 [B, max_len, P, D]
            text_tokens_batch = torch.stack([pad_1d(t, pad_value=0) for t in text_tokens_list], dim=0)
            text_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in text_mask_list], dim=0)
            audio_feats_batch = torch.stack([pad_3d(f) for f in audio_feats_list], dim=0)
            audio_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in audio_mask_list], dim=0)
            loss_mask_batch = torch.stack([pad_1d(m, pad_value=0) for m in loss_mask_list], dim=0)
            labels_batch = torch.stack([pad_1d(lbl, pad_value=0) for lbl in labels_list], dim=0)

            # Position ids: [B, T], simple 0..L_i-1 then padded with 0
            # 位置 id：[B, T]，有效部分为 0..L_i-1，填充部分补 0
            position_ids_list = []
            for L in lengths:
                L_clip = min(L, max_len)
                pos = torch.arange(0, L_clip, device=device)
                if L_clip < max_len:
                    pad = torch.zeros(max_len - L_clip, dtype=pos.dtype, device=device)
                    pos = torch.cat([pos, pad], dim=0)
                position_ids_list.append(pos)
            position_ids = torch.stack(position_ids_list, dim=0)
        else:
            # Empty batch fallback (shouldn't really happen)
            # 空批兜底（正常情况下不应出现）
            text_tokens_batch = torch.zeros((0, self.max_len), dtype=torch.int32, device=device)
            text_mask_batch = torch.zeros_like(text_tokens_batch)
            audio_feats_batch = torch.zeros(
                (0, self.max_len, self.patch_size, self.feat_dim), dtype=torch.float32, device=device
            )
            audio_mask_batch = torch.zeros_like(text_tokens_batch)
            loss_mask_batch = torch.zeros_like(text_tokens_batch)
            labels_batch = torch.zeros_like(text_tokens_batch)
            position_ids = torch.zeros_like(text_tokens_batch)

        # 返回模型 forward 所需的全部张量
        return {
            "text_tokens": text_tokens_batch,
            "audio_feats": audio_feats_batch,
            "text_mask": text_mask_batch,
            "audio_mask": audio_mask_batch,
            "loss_mask": loss_mask_batch,
            "position_ids": position_ids,
            "labels": labels_batch,
        }

    # ------------------------------------------------------------------ #
    # Feature extraction helpers
    # 特征提取辅助方法
    # ------------------------------------------------------------------ #
    def extract_audio_feats(self, audio_data: torch.Tensor):
        """把波形编码为潜变量特征，并按 patch_size 分组为 [B, T, P, D]。

        每个序列位置对应 P = patch_size 帧 VAE 特征，D 为特征维度。
        """
        audio_feats = self.encode_audio(audio_data)
        # VAE 帧数不能被 patch_size 整除时，在时间维末尾补零对齐
        if audio_feats.size(1) % self.patch_size != 0:
            audio_feats_ = audio_feats.transpose(1, 2)
            padding = nn.functional.pad(audio_feats_, (0, self.patch_size - audio_feats.size(1) % self.patch_size))
            audio_feats = padding.transpose(1, 2)

        # 把 [B, T*P, D] 重排为 [B, T, P, D]：T 个序列位置，每个位置含 P 帧特征
        audio_feats = rearrange(audio_feats, "b (t p) c -> b t p c", p=self.patch_size)
        return audio_feats

    def process_tts_data(self, audio_token: torch.Tensor, text_token: torch.Tensor):
        """对单个样本执行序列打包：拼接文本段与音频段，生成输入 token、特征及各类 mask。

        Args:
            audio_token: 去填充后的音频波形（一维张量）。
            text_token: 去填充后的文本 token id 序列。

        Returns:
            (text_token_info, audio_feat_info, text_mask, audio_mask, loss_mask, labels)
            六个张量，长度均为 文本长度 + 音频长度 + 1。
        """
        # 文本段末尾拼接音频起始特殊 token（101）
        text_token_info = torch.cat(
            [
                text_token,
                torch.tensor(
                    [self.audio_start_id],
                    dtype=torch.int32,
                    device=text_token.device,
                ),
            ],
            dim=-1,
        )
        text_length = text_token_info.shape[0]  # 文本长度（含末尾的 101）
        # 提取音频特征：[1, T, P, D] -> [T, P, D]
        audio_feat_info = self.extract_audio_feats(audio_token).squeeze(0)
        audio_length = audio_feat_info.shape[0]

        # 音频段对应的 token 位置用 0 占位，最后拼接音频结束特殊 token（102）
        text_pad_token = torch.zeros(audio_length, dtype=torch.int32, device=text_token.device)
        text_token_info = torch.cat(
            [
                text_token_info,
                text_pad_token,
                torch.tensor([self.audio_end_id], dtype=torch.int32, device=text_token.device),
            ]
        )
        # 文本段（含 101）与结尾 102 位置上的特征用零块占位，
        # 使特征序列与 token 序列等长、位置一一错位对应（用于在当前位置预测下一位置的特征）
        audio_pad_feat = torch.zeros(
            (text_length, self.patch_size, audio_feat_info.size(-1)),
            dtype=torch.float32,
            device=text_token.device,
        )
        audio_feat_info = torch.cat([audio_pad_feat, audio_feat_info, audio_pad_feat[0:1, ...]], dim=0)

        # text_mask：文本段（含 101）与结尾 102 位置为 1，音频段为 0
        text_mask = (
            torch.cat([torch.ones(text_length), torch.zeros(audio_length), torch.ones(1)])
            .type(torch.int32)
            .to(text_token.device)
        )
        # audio_mask：仅音频段为 1，其余（文本段与结尾位置）为 0
        audio_mask = (
            torch.cat([torch.zeros(text_length), torch.ones(audio_length), torch.zeros(1)])
            .type(torch.int32)
            .to(text_token.device)
        )
        # loss_mask：仅音频段为 1 —— 损失（diffusion + stop）只在音频段计算
        loss_mask = (
            torch.cat([torch.zeros(text_length), torch.ones(audio_length), torch.zeros(1)])
            .type(torch.int32)
            .to(text_token.device)
        )

        # stop 标签：仅在倒数第二个位置（最后一个音频位置）为 1，标记音频结束
        labels = torch.zeros(text_length + audio_length + 1).type(torch.int32).to(text_token.device)
        labels[-2] = 1

        return (
            text_token_info,
            audio_feat_info,
            text_mask,
            audio_mask,
            loss_mask,
            labels,
        )
