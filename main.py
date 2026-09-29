from voxcpm import VoxCPMTTS
import soundfile as sf
import time
model = VoxCPMTTS(
    voxcpm_model_path="/data/models/VoxCPM-0.5B",
    optimize=False
)

text = '太阳一晒，我就变成气升到天空，我又变成无数极小极小的点儿。连成一片在空中漂浮，有时候我穿着白衣服，有时候我穿着黑衣服。早晨和傍晚，我又把红袍披在身上。人们叫我云，气温越来越低。变成了无数小水滴，小水滴聚在一起落下来。人们叫我雨，有时候我变成小硬球。'

PROMPT_PATH = './assets/zero_shot_prompt.wav'
PROMPT_TEXT = '希望你以后能够做得比我还好呦。'

start_time = time.time()

wav = model.generate(
    text=text,
    cfg_value=2.0,
    prompt_wav_path=PROMPT_PATH,
    prompt_text=PROMPT_TEXT,
    inference_timesteps=10,
    seed=42,
)

print(f"生成完成，总耗时 {time.time() - start_time:.2f}s")