# VoxCPM 模型模块的通用工具集：音频加载、随机种子管理、中文分词包装、
# dtype/设备解析等辅助函数。
import os
from typing import List, Optional
import torch
import soundfile as sf
from transformers import PreTrainedTokenizer


def load_audio(path: str, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """加载音频文件为 float32 单声道张量，形状为 (channels, time)。

    直接使用 soundfile 读取（torchaudio >= 2.10 把编解码工作委托给了
    torchcodec，我们避免引入该依赖）。可选地重采样到 ``target_sr``。

    Args:
        path: 音频文件路径。
        target_sr: 目标采样率；``None`` 表示保持原始采样率。

    Returns:
        tuple[torch.Tensor, int]: (音频张量 (1, time)，实际采样率)。
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(wav.T)  # (channels, time)
    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)  # 多声道取平均，混合为单声道
    if target_sr is not None and sr != target_sr:
        import torchaudio

        audio = torchaudio.functional.resample(audio, sr, target_sr)  # 重采样到目标采样率
        sr = target_sr
    return audio, sr

# 被视为低精度的 dtype 名称集合（MPS 上这些精度不稳定，会被强制回退 float32）
_LOW_PRECISION_DTYPES = {"bfloat16", "bf16", "float16", "fp16"}
# VOXCPM_MPS_DTYPE 环境变量允许的取值集合
_VALID_DTYPE_OVERRIDES = {
    "bfloat16",
    "bf16",
    "float16",
    "fp16",
    "float32",
    "fp32",
}


# Ref: https://github.com/OpenBMB/VoxCPM/issues/256#issuecomment-4235252732
# Explicitly close partially-consumed generators so inference_mode cleanup
# does not get deferred to Python's GC/finalizer path.
# 取出生成器的首个结果后立即显式关闭生成器，
# 避免 inference_mode 的清理动作被推迟到 Python 的 GC/终结器路径。
def consume_single_result(gen):
    """从生成器中取出第一个值，然后关闭生成器。

    用于"一次性"消费生成器的场景：调用方只需要生成器的第一个结果，
    之后不希望生成器继续占用资源，因此主动关闭它。

    Args:
        gen: 生成器对象（通常只会有一个结果，或者调用方只需要第一个）。

    Returns:
        生成器 yield 出的第一个值。
    """
    result = next(gen)      # 取出第一个值
    gen.close()             # 立即关闭，释放资源
    return result


def materialize_generation_seed(seed: Optional[int]) -> int:
    """为一次生成请求确定具体的随机种子。

    Args:
        seed: 用户指定的种子；``None`` 表示从全局随机状态派生一个。

    Returns:
        int: 具体可用的整数随机种子。
    """
    if seed is not None:
        return int(seed)
    return int(torch.seed() & 0xFFFFFFFF)  # 取全局种子的低 32 位作为本次生成种子


def apply_generation_seed(seed: int) -> None:
    """把随机种子写入 torch 全局随机状态（含所有 CUDA 设备），保证生成可复现。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """创建一个把多字中文词切分为单字的分词器包装器。

    本函数基于给定的分词器创建一个包装器，自动把多字中文词拆成单个汉字，
    用于保证中文文本分词粒度的一致性。

    Args:
        tokenizer: 被包装的基础分词器。

    Returns:
        处理多字中文词的 CharTokenizerWrapper 实例。

    Example:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # Pre-compute multi-character tokens (length >= 2, pure Chinese characters)
    # 预先计算多字词集合（长度 >= 2 且全部为汉字的词元）
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """处理多字中文词的分词器包装类。

        该包装器自动把多字中文词拆分为单个汉字，
        同时保持原始分词器的接口不变。
        """

        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """用基础分词器初始化包装器。

            Args:
                base_tokenizer: 被包装的分词器。
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """分词并把多字中文词拆分为单个汉字。

            Args:
                text: 待分词的输入文本。
                **kwargs: 透传给基础分词器的额外参数。

            Returns:
                处理后的词元列表（多字中文词已被拆分）。

            Example:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # Returns ["你", "好", "世", "界"] instead of ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")

            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []

            for token in tokens:
                # Remove possible subword prefix
                clean_token = token.replace("▁", "")  # 去除 SentencePiece 的子词前缀 ▁

                if clean_token in self.multichar_tokens:
                    # Split multi-character token into single characters
                    chars = list(clean_token)  # 命中多字中文词：拆成单字
                    processed.extend(chars)
                else:
                    processed.append(token)

            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """调用分词器并返回 token id 列表。

            该方法提供与原始分词器一致的调用接口，
            但额外处理了多字中文词的拆分。

            Args:
                text: 待分词的输入文本。
                **kwargs: 透传给基础分词器的额外参数。

            Returns:
                token id 列表。

            Raises:
                TypeError: 输入不是字符串。
                ValueError: 分词失败。
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def get_dtype(dtype: str):
    """把 dtype 名称字符串转换为对应的 torch 数据类型对象。

    Args:
        dtype: dtype 名称（如 "bfloat16"/"fp16"/"float32"）。

    Returns:
        torch.dtype: 对应的 torch 数据类型。

    Raises:
        ValueError: dtype 名称不受支持。
    """
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _has_mps() -> bool:
    """检查当前环境是否有可用的 Apple MPS 后端。"""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def pick_runtime_dtype(device: str, configured_dtype: str) -> str:
    """为已解析的设备选择一个安全的运行时精度。

    在 Apple Silicon（MPS）上，bfloat16/float16 会在扩散自回归循环中
    产生明显的数值漂移，导致输出毛刺、模型的坏例检测器触发无限重试。
    目前 float32 是唯一稳定的选择；CUDA 与 CPU 保持 checkpoint 训练时的精度。

    用户可用 ``VOXCPM_MPS_DTYPE``（例如 ``bfloat16``）覆盖，
    以便测试未来 MPS 的改进。

    Args:
        device: 已解析的运行设备名（如 "cuda"/"mps"/"cpu"）。
        configured_dtype: 配置/checkpoint 中声明的精度名称。

    Returns:
        str: 实际应使用的精度名称。
    """
    if device != "mps":
        return configured_dtype

    # MPS：优先读取环境变量覆盖
    override = os.environ.get("VOXCPM_MPS_DTYPE", "").strip().lower()
    if override:
        if override not in _VALID_DTYPE_OVERRIDES:
            raise ValueError(f"VOXCPM_MPS_DTYPE='{override}' is not one of " f"{sorted(_VALID_DTYPE_OVERRIDES)}")
        return override

    # 配置为低精度时强制回退 float32，避免数值漂移导致生成异常
    if (configured_dtype or "").lower() in _LOW_PRECISION_DTYPES:
        return "float32"
    return configured_dtype


def auto_select_device(preferred_device: Optional[str] = "cuda") -> str:
    """自动选择运行设备。

    偏好顺序：
    - 若首选设备可用，直接使用它
    - 否则按 CUDA -> MPS -> CPU 的顺序回退

    Args:
        preferred_device: 首选设备名，``None`` 时默认 "cuda"。

    Returns:
        str: 实际可用的设备名。
    """
    preferred = (preferred_device or "cuda").strip().lower()

    if preferred.startswith("cuda") and torch.cuda.is_available():
        return preferred
    if preferred == "mps" and _has_mps():
        return "mps"
    if preferred == "cpu":
        return "cpu"

    # 首选设备不可用：按 CUDA -> MPS -> CPU 顺序回退
    if torch.cuda.is_available():
        return "cuda"
    if _has_mps():
        return "mps"
    return "cpu"


def resolve_runtime_device(device: Optional[str], configured_device: str = "cuda") -> str:
    """解析实际的运行设备。

    语义：
    - ``device`` 为 ``None`` 或 ``"auto"``：使用自动回退选择
    - 其他情况：视为用户的显式选择，并校验其可用性

    Args:
        device: 用户传入的设备描述（None/"auto"/"cuda"/"cuda:0"/"mps"/"cpu"）。
        configured_device: 自动选择时的首选设备。

    Returns:
        str: 实际使用的设备名。

    Raises:
        ValueError: 显式指定的设备不可用，或设备名不受支持。
    """
    explicit = None if device is None else device.strip().lower()

    if explicit is None or explicit == "auto":
        return auto_select_device(configured_device)

    # 显式指定的设备：逐一校验可用性，不可用则直接报错（而非静默回退）
    if explicit.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested device '{device}', but CUDA is not available. " "Use device='auto' for automatic fallback."
            )
        return explicit
    if explicit == "mps":
        if not _has_mps():
            raise ValueError(
                "Requested device 'mps', but MPS is not available. " "Use device='auto' for automatic fallback."
            )
        return "mps"
    if explicit == "cpu":
        return "cpu"

    raise ValueError(
        f"Unsupported device '{device}'. Supported values are 'auto', 'cpu', 'mps', "
        "'cuda', or indexed CUDA devices like 'cuda:0'."
    )
