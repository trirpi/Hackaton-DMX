import torch
import torchaudio
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import time

# Load MusicGen model & processor from Hugging Face
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
processor = AutoProcessor.from_pretrained("facebook/musicgen-large")

# Set up input prompt
text_prompt = ["A relaxing jazz piece with soft piano and saxophone"]
text_prompt= ["Generate a happy electronic melody"]

# Tokenize input text
inputs = processor(text=text_prompt, padding=True, return_tensors="pt")

# Generate waveform (no need for decode step!)
start_time = time.time()
with torch.no_grad():
    audio_waveform = model.generate(**inputs, do_sample=True, max_new_tokens=200)
end_time = time.time()
print(f"Generation took {end_time - start_time:.2f} seconds")



# Extract tensor and sample rate
waveform = audio_waveform[0]  # (1, num_samples)
sample_rate = model.config.audio_encoder.sampling_rate

# Save the generated music
torchaudio.save("generated_music.wav", waveform, sample_rate)
