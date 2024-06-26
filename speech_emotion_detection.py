import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.wav2vec.models import Wav2Vec2ForSpeechClassification
from pathlib import Path
from threading import Event
import pyaudio
import wave
import tqdm
import os
import subprocess

dict_emotions_ravdess = {
    0: 'Neutral',
    1: 'Calm',
    2: 'Happy',
    3: 'Sad',
    4: 'Angry',
    5: 'Fear',
    6: 'Disgust',
    7: 'Surprise'
}

audio_name_or_path = "trained_model\\wav2vec"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(audio_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(audio_name_or_path)
sampling_rate = feature_extractor.sampling_rate

# for wav2vec
model = Wav2Vec2ForSpeechClassification.from_pretrained(audio_name_or_path).to(device)

path_record = "Record.wav"

#RAVDESS Dataset (https://zenodo.org/records/1188976)
path_test_audio = "test\\audio"
path_test_video = "test\\video"

def convert_mp4_to_wav(path):

    path_save = os.path.join(path_test_video, path.name[:-3] + "wav")
    if not os.path.exists(path_save):
        ff_audio = "ffmpeg -i {} -vn -acodec libmp3lame -ar 16000 -ac 2 {}".format(
            path, path_save
        )
        subprocess.call(ff_audio, shell=True)

    return path_save

def speech_file_to_array_fn(path, sampling_rate):
    speech_array, _sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(_sampling_rate, sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech


def predict(path, sampling_rate):
    speech = speech_file_to_array_fn(path, sampling_rate)
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    inputs = {key: inputs[key].to(device) for key in inputs}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    outputs = [{"Emotion": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in
               enumerate(scores)]
    return outputs

def recordAudio(stop_event):
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt32  # 16 bits per sample
    channels = 1
    fs = 16000  # Record at 16000Hz
    seconds = 4
    filename = "Record.wav"

    while not stop_event.is_set():
        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 10 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')

        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        outputs = predict(path_record, sampling_rate)
        print(outputs)

def testAudios():
    for path in Path(path_test_audio).glob("**/*.wav"):
        name = str(path).split('/')[-1].split('.')[0]
        label = dict_emotions_ravdess[int(name.split("-")[2]) - 1]  # Start emotions in 0
        print("Audio label is: ", label)
        actor = int(name.split("-")[-1])
        outputs = predict(path, sampling_rate)
        print(outputs)

def audioFromVideo():
    for path in Path(path_test_video).glob("**/*.mp4"):
        name = str(path).split('/')[-1].split('.')[0]
        label = dict_emotions_ravdess[int(name.split("-")[2]) - 1]  # Start emotions in 0
        print("Audio label is: ", label)
        actor = int(name.split("-")[-1])
        wav_path = convert_mp4_to_wav(path)
        outputs = predict(wav_path, sampling_rate)
        print(outputs)

# DEBUG MODE
# while True:
#     choice = int(input("Enter 1 to predict your own record \nEnter 2 to test recorded audios. \nEnter 3 to test audios from recored videos. \nEnter 4 to quit. \n"))
#     if choice == 1:
#         recordAudio()
#     elif choice == 2:
#         testAudios()
#     elif choice == 3:
#         audioFromVideo()
#     else:
#         quit()