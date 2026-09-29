"""
VoxCPM1 (0.5B) 语音合成（TTS）流水线：模型加载 + 高层语音生成 API。

从 OpenBMB/VoxCPM 精简而来：无 LoRA、无去噪器、单一架构。
"""

import os
import re
import sys
import numpy as np
from typing import Generator, Optional

from .model.voxcpm import VoxCPMModel


class VoxCPMTTS:
    """VoxCPM1 语音合成高层封装：负责模型加载与面向用户的生成接口。"""

    def __init__(
        self,
        voxcpm_model_path: str,
        optimize: bool = True,
        device: str | None = None,
    ):
        """初始化 VoxCPM1 语音合成流水线。

        Args:
            voxcpm_model_path: VoxCPM 模型资源（权重、配置等）的本地文件系统路径。
            optimize: 是否使用 torch.compile 优化模型。
            device: 运行设备。若设为 ``None`` 或 ``"auto"``，
                VoxCPM 会自动选择（优先 CUDA，其次 MPS，最后 CPU）。
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}", file=sys.stderr)
        self.tts_model = VoxCPMModel.from_local(
            voxcpm_model_path,
            optimize=optimize,
            device=device,
        )
        self.text_normalizer = None  # 惰性加载（首次调用 normalize 时才初始化）

        if optimize:
            # torch.compile 首次执行需要触发编译，这里跑一次 dummy 生成做预热
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """一次性（非流式）合成完整音频。

        Args:
            text: 待合成的输入文本。
            prompt_wav_path: 用于声音克隆的提示音频路径（必须与 ``prompt_text`` 成对提供）。
            prompt_text: 提示音频对应的文本内容。
            cfg_value: 无分类器引导（CFG）强度。
            inference_timesteps: 推理步数（流匹配 ODE 求解步数）。
            min_len: 音频 token 长度下限。
            max_len: 生成过程中 token 长度上限。
            normalize: 生成前是否执行文本正则化。
            retry_badcase: 是否启用坏例重试。
            seed: 可选随机种子，用于结果复现。

        Returns:
            np.ndarray: 一维 float32 波形数组（CPU）。
        """
        text, prompt_cache = self._prepare_generation(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            normalize=normalize,
        )
        # 模型层非流式接口：内部完成推理并直接返回 (wav, text_token, feat) 三元组
        wav, _, _ = self.tts_model.generate_with_prompt_cache(
            target_text=text,
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            seed=seed,
        )
        return wav.squeeze(0).cpu().numpy()

    def generate_streaming(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 4096,
        normalize: bool = False,
        retry_badcase: bool = True,
        retry_badcase_max_times: int = 3,
        retry_badcase_ratio_threshold: float = 6.0,
        seed: Optional[int] = None,
    ) -> Generator[np.ndarray, None, None]:
        """流式合成音频，逐块 yield 已解码的波形片段。

        Args:
            text: 待合成的输入文本。
            prompt_wav_path: 用于声音克隆的提示音频路径（必须与 ``prompt_text`` 成对提供）。
            prompt_text: 提示音频对应的文本内容。
            cfg_value: 无分类器引导（CFG）强度。
            inference_timesteps: 推理步数（流匹配 ODE 求解步数）。
            min_len: 音频 token 长度下限。
            max_len: 生成过程中 token 长度上限。
            normalize: 生成前是否执行文本正则化。
            retry_badcase: 是否启用坏例重试。
            seed: 可选随机种子，用于结果复现。

        Yields:
            np.ndarray: 每块一维 float32 波形数组（CPU）。
        """
        text, prompt_cache = self._prepare_generation(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            normalize=normalize,
        )
        gen = self.tts_model.generate_with_prompt_cache_streaming(
            target_text=text,
            prompt_cache=prompt_cache,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            retry_badcase=retry_badcase,
            retry_badcase_max_times=retry_badcase_max_times,
            retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
            seed=seed,
        )
        try:
            for wav, _, _ in gen:
                yield wav.squeeze(0).cpu().numpy()
        finally:
            # 调用方提前中断消费时，显式关闭模型侧生成器，
            # 避免 inference_mode 的清理被推迟到 GC（见 model/utils.py）
            gen.close()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _prepare_generation(
        self,
        text: str,
        prompt_wav_path: str = None,
        prompt_text: str = None,
        normalize: bool = False,
    ) -> tuple[str, Optional[dict]]:
        """生成前的公共预处理：输入校验、文本清洗、提示缓存构建、文本归一化。

        普通方法（非生成器），调用即执行，返回模型生成接口所需的直接输入。

        Args:
            text: 待合成的输入文本。
            prompt_wav_path: 提示音频路径（与 ``prompt_text`` 成对提供，或均为 None）。
            prompt_text: 提示音频对应的文本内容。
            normalize: 是否执行文本正则化。

        Returns:
            tuple[str, Optional[dict]]: (处理后的目标文本, 提示缓存)。
                零样本（未提供提示音频）时提示缓存为 None。
        """
        # ---- 输入校验 --------------------------------------------------------
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        # ---- 文本预处理 ------------------------------------------------------
        # 换行替换为空格，再压缩连续空白为单个空格
        text = re.sub(r"\s+", " ", text.replace("\n", " "))

        # ---- 提示缓存（用于声音克隆） ----------------------------------------
        prompt_cache = None
        if prompt_wav_path is not None:
            # 预先编码提示音频，构建提示缓存（文本 + 音频隐变量特征），
            # 后续生成时直接复用，避免每次重新编码参考音频
            prompt_cache = self.tts_model.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_wav_path,
            )

        # ---- 文本归一化 ------------------------------------------------------
        if normalize:
            if self.text_normalizer is None:
                from .utils.text_normalize import TextNormalizer
                self.text_normalizer = TextNormalizer()
            text = self.text_normalizer.normalize(text)

        return text, prompt_cache
