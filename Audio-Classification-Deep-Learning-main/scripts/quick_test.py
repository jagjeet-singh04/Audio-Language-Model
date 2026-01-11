import os, sys
project_dir = r"C:\Users\jagje\Downloads\Audio-Classification-Deep-Learning-main\Audio-Classification-Deep-Learning-main"
sys.path.insert(0, project_dir)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AudioClassification.settings")
import django
django.setup()

from AudioClassification.functions import ANN_print_prediction, CNN1D_print_prediction, CNN2D_print_prediction
import numpy as np
import soundfile as sf

sr = 22050
seconds = 2
samples = seconds * sr
# Generate a quiet 440Hz sine wave

t = np.linspace(0, seconds, samples, endpoint=False)
tone = 0.2 * np.sin(2 * np.pi * 440 * t)

media_dir = os.path.join(project_dir, "media")
os.makedirs(media_dir, exist_ok=True)
wav_path = os.path.join(media_dir, "test_tone.wav")

sf.write(wav_path, tone, sr)

print("WAV written:", wav_path)
print("ANN:", ANN_print_prediction(wav_path))
print("CNN1D:", CNN1D_print_prediction(wav_path))
print("CNN2D:", CNN2D_print_prediction(wav_path))
