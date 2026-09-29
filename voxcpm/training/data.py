"""VoxCPM 训练数据流水线：加载 jsonl 清单数据集、惰性解码音频（单声道 + 重采样）、
批拼接（collate），以及交给序列打包器（packer）前的批处理封装。"""
import math
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset

from ..model.utils import load_audio
from ..model.voxcpm import VoxCPMConfig
from ..modules.audiovae import AudioVAE
from .packers import AudioFeatureProcessingPacker

# 清单文件中默认的文本列名与音频（路径）列名
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_AUDIO_COLUMN = "audio"


def load_audio_text_datasets(
    train_manifest: str,
    val_manifest: str = "",
    text_column: str = DEFAULT_TEXT_COLUMN,
    audio_column: str = DEFAULT_AUDIO_COLUMN,
    sample_rate: int = 16_000,
    num_proc: int = 1,
) -> Tuple[Dataset, Optional[Dataset]]:
    """加载包含 ``text`` 与 ``audio``（音频文件路径）两列的训练/验证 jsonl 清单文件。

    此处音频仍以文件路径形式保存，真正的解码（转单声道、重采样到
    ``sample_rate``）延迟到 ``HFVoxCPMDataset`` 中通过 soundfile 惰性完成。

    Args:
        train_manifest: 训练集 jsonl 清单文件路径。
        val_manifest: 验证集 jsonl 清单文件路径，传空字符串表示不加载验证集。
        text_column: 清单中文本所在的列名。
        audio_column: 清单中音频路径所在的列名。
        sample_rate: 目标采样率（Hz）。
        num_proc: 预留的并行加载进程数（当前实现未使用）。

    Returns:
        (train_ds, val_ds) 二元组：训练数据集与验证数据集；未提供验证清单时为 None。
    """
    # 组装数据文件映射：train 必选，validation 可选
    data_files = {"train": train_manifest}
    if val_manifest:
        data_files["validation"] = val_manifest

    # 以 json 格式加载清单，得到含 train/validation 子集的 DatasetDict
    dataset_dict: DatasetDict = load_dataset("json", data_files=data_files)

    def prepare(ds: Dataset) -> Dataset:
        # 统一列名：校验音频列存在，并把自定义列名重命名为默认的 audio/text
        if audio_column not in ds.column_names:
            raise ValueError(f"Expected '{audio_column}' column in manifest.")
        if audio_column != DEFAULT_AUDIO_COLUMN:
            ds = ds.rename_column(audio_column, DEFAULT_AUDIO_COLUMN)
        if text_column != DEFAULT_TEXT_COLUMN:
            ds = ds.rename_column(text_column, DEFAULT_TEXT_COLUMN)
        return ds

    # 分别对训练集与验证集做列名校正；未提供验证清单时返回 None
    train_ds = prepare(dataset_dict["train"])
    val_ds = prepare(dataset_dict["validation"]) if "validation" in dataset_dict else None
    return train_ds, val_ds


def compute_sample_lengths(
    ds: Dataset,
    audio_vae_fps: float = 25,
    patch_size: int = 2,
) -> List[int]:
    """
    预估每个样本经过 packer 之后的大致序列长度（text+audio），用于过滤超长样本。

    逻辑与 AudioFeatureProcessingPacker 一致：
    - 文本长度: len(text_ids)
    - 音频长度: duration(s) * audio_vae_fps -> 近似 VAE 帧数 t_vae; t_seq = ceil(t_vae / patch_size)
    - 总长度: text_len + t_seq + 2 (101/102)
    """
    # 每个样本文本 token 序列的长度
    text_lens = [len(t) for t in ds["text_ids"]]

    lengths = []
    for text_len, audio_path in zip(text_lens, ds[DEFAULT_AUDIO_COLUMN]):
        # 只读音频文件头信息（不解码波形），按 总帧数/采样率 估算时长（秒）
        info = sf.info(audio_path)
        duration = info.frames / float(info.samplerate)
        # 时长 × VAE 帧率 -> 近似 VAE 潜变量帧数；再按 patch_size 折算为序列帧数
        t_vae = math.ceil(duration * audio_vae_fps)
        t_seq = math.ceil(t_vae / patch_size)
        # 总长度 = 文本长度 + 音频序列长度 + 2 个特殊 token（101/102）
        lengths.append(text_len + t_seq + 2)

    return lengths


class HFVoxCPMDataset(TorchDataset):
    """
    对 jsonl 加载的 HuggingFace 数据集的轻量封装，返回 PyTorch 友好的样本。
    音频通过 soundfile 解码（单声道，并重采样到 ``sample_rate``）。

    Args:
        dataset: 已含 text_ids 与 audio（路径）列的 HuggingFace 数据集。
        sample_rate: 音频解码后的目标采样率（Hz）。
    """

    def __init__(self, dataset: Dataset, sample_rate: int = 16_000):
        self.dataset = dataset
        self.sample_rate = sample_rate

    def __len__(self):
        """返回数据集的样本总数。"""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """按索引取样本：惰性解码音频（单声道 + 重采样），返回 text_ids 与音频波形。"""
        item = self.dataset[idx]
        # 读取音频文件并重采样到目标采样率，得到单声道波形
        audio, _ = load_audio(item[DEFAULT_AUDIO_COLUMN], target_sr=self.sample_rate)
        return {
            "text_ids": item["text_ids"],
            # 去掉多余的通道维后转为 numpy 数组
            "audio_array": audio.squeeze(0).numpy(),
        }

    @staticmethod
    def pad_sequences(seqs: List[torch.Tensor], pad_value: float):
        """把一组一维序列填充到批内最大长度后堆叠为 [B, max_len]，填充值为 ``pad_value``。"""
        # 空列表直接返回空张量
        if not seqs:
            return torch.empty(0)
        # 批内最长序列的长度，即统一填充的目标长度
        max_len = max(seq.shape[0] for seq in seqs)
        padded = []
        for seq in seqs:
            # 短于最大长度的序列在末尾补 pad_value
            if seq.shape[0] < max_len:
                pad_width = (0, max_len - seq.shape[0])
                seq = torch.nn.functional.pad(seq, pad_width, value=pad_value)
            padded.append(seq)
        return torch.stack(padded)

    @classmethod
    def collate_fn(cls, batch: List[Dict]):
        """批拼接（collate）：把样本列表组织成填充后的 text_tokens / audio_tokens 两个批张量。

        填充值统一使用 -100，便于后续 packer 通过 -100 定位有效长度并去除填充。
        """
        # 文本 token id 转 int32 张量，音频波形转 float32 张量
        text_tensors = [torch.tensor(sample["text_ids"], dtype=torch.int32) for sample in batch]
        audio_tensors = [torch.tensor(sample["audio_array"], dtype=torch.float32) for sample in batch]

        # 分别填充到批内最大长度（填充值 -100）
        text_padded = cls.pad_sequences(text_tensors, pad_value=-100)
        audio_padded = cls.pad_sequences(audio_tensors, pad_value=-100.0)

        return {
            "text_tokens": text_padded,
            "audio_tokens": audio_padded,
        }


class BatchProcessor:
    """
    对 ``AudioFeatureProcessingPacker`` 的封装：训练循环只需把批拼接后的
    batch 直接喂给它，即可完成音频特征提取与序列打包。

    Args:
        config: VoxCPM 模型配置（提供 max_length / patch_size / feat_dim）。
        audio_vae: 用于音频特征提取的 AudioVAE 模块。
        device: 计算设备（如 cuda）。
    """

    def __init__(
        self,
        *,
        config: VoxCPMConfig,
        audio_vae: AudioVAE,
        device: torch.device,
    ):
        self.device = device
        self.audio_vae = audio_vae
        # 将 AudioVAE 移到指定设备（音频特征提取在其上进行）
        self.audio_vae.to(device)
        # 构造序列打包器（packer）：负责音频特征提取、文本/音频拼接与 mask 生成
        self.packer = AudioFeatureProcessingPacker(
            max_len=config.max_length,
            patch_size=config.patch_size,
            feat_dim=config.feat_dim,
            audio_vae=self.audio_vae,
        )

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """把批数据搬到计算设备后交给 packer，返回模型 forward 所需的打包结果。"""
        return self.packer(
            audio_tokens=batch["audio_tokens"].to(self.device),
            text_tokens=batch["text_tokens"].to(self.device),
        )


def build_dataloader(
    hf_dataset: Dataset,
    *,
    accelerator,
    batch_size: int,
    num_workers: int,
    sample_rate: int = 16_000,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    """构建训练用数据加载器。

    Args:
        hf_dataset: 已含 text_ids 与 audio（路径）列的 HuggingFace 数据集。
        accelerator: 训练加速器；多卡时负责为数据加载器挂载分布式采样器。
        batch_size: 每个批的样本数。
        num_workers: 数据加载的工作进程数。
        sample_rate: 音频解码的目标采样率（Hz）。
        drop_last: 批内样本数不足 batch_size 时是否丢弃最后一个不完整批。

    Returns:
        可直接迭代出 text_tokens / audio_tokens 批数据的数据加载器。
    """
    torch_dataset = HFVoxCPMDataset(hf_dataset, sample_rate=sample_rate)
    # Standard padding-based batching; Accelerator will attach DistributedSampler if needed.
    # 基于填充（padding）的标准批处理；多卡时 Accelerator 会挂载分布式采样器（DistributedSampler）。
    return accelerator.prepare_dataloader(
        torch_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=HFVoxCPMDataset.collate_fn,
        drop_last=drop_last,
    )
