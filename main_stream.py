"""流式语音合成演示：边生成边播放（无需等待完整音频生成完毕）。"""

import time
import soundfile as sf
import numpy as np

from voxcpm import VoxCPMTTS

# 加载模型
model = VoxCPMTTS(
    voxcpm_model_path="/data/models/VoxCPM-0.5B",
    optimize=False,
)

text = (
    "太阳一晒，我就变成气升到天空，我又变成无数极小极小的点儿。"
    "连成一片在空中漂浮，有时候我穿着白衣服，有时候我穿着黑衣服。"
    "早晨和傍晚，我又把红袍披在身上。人们叫我云，气温越来越低。"
    "变成了无数小水滴，小水滴聚在一起落下来。人们叫我雨，有时候我变成小硬球。"
)

PROMPT_PATH = "./assets/zero_shot_prompt.wav"
PROMPT_TEXT = "希望你以后能够做得比我还好呦。"

# 收集所有片段，统计流式输出
all_chunks: list[np.ndarray] = []
chunk_count = 0
start_time = time.time()

print("开始流式生成...")
for wav_chunk in model.generate_streaming(
    text=text,
    cfg_value=2.0,
    prompt_wav_path=PROMPT_PATH,
    prompt_text=PROMPT_TEXT,
    inference_timesteps=10,
    seed=42,
):
    chunk_count += 1
    all_chunks.append(wav_chunk)
    print(f"  收到第 {chunk_count} 个音频片段，当前总长度: {sum(len(c) for c in all_chunks)} 采样点")

total_time = time.time() - start_time
full_wav = np.concatenate(all_chunks)

print(f"生成完成，共 {chunk_count} 个片段，总耗时 {total_time:.2f}s")
print(f"音频总长度: {len(full_wav)} 采样点，采样率 16000Hz，时长约 {len(full_wav)/16000:.1f}s")

# 可选：保存到文件
output_path = "output_streaming.wav"
sf.write(output_path, full_wav, 16000)
print(f"已保存至 {output_path}")
