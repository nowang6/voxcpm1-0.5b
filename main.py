from voxcpm import VoxCPMTTS
import soundfile as sf
import time
model = VoxCPMTTS(
    voxcpm_model_path="/data/models/VoxCPM-0.5B",
    optimize=False
)

text = '一次列一百个单词不方便记忆哦，我可以每次给你分享一些常用的含s的单词，你慢慢积累会更高效，需要继续再给你补充更多单词哦。'

PROMPT_PATH = './assets/gentle_senior.wav'
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


sf.write('output.wav', wav, 16000)
print(f"生成完成，总耗时 {time.time() - start_time:.2f}s")