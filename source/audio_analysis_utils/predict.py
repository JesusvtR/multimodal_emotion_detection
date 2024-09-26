import source.audio_analysis_utils.utils as utils
import source.config as config
import source.audio_analysis_utils.preprocess_data as data
import source.audio_analysis_utils.transcribe_audio as transcribe_audio

import torch
import numpy as np
import pyaudio

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import queue
import speech_recognition as sr
import io
import soundfile as sf


# Record parameters 
DURATION = 30  # Duration in seconds %RS_3.3%

def predict(result_queue):
    best_audio_hyperparameters = utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
    AUDIO_SAMPLE_RATE = 16000  # Sampling rate in Hz
    # Record audio with microphone
    recorder = sr.Recognizer()
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
        print("\nRecording...\n")
        audio_data = recorder.listen(source, timeout=DURATION)
        
    wav_bytes = audio_data.get_wav_data(convert_rate=AUDIO_SAMPLE_RATE)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    
    if config.SAVE_AUDIO:
        # Guardar en un archivo WAV
        wav.write(config.INPUT_FOLDER_PATH + 'output.wav', AUDIO_SAMPLE_RATE, audio_data)

        audio_file = utils.find_filename_match('output', config.INPUT_FOLDER_PATH)
        audio_file_only = audio_file.split(config.INPUT_FOLDER_PATH)[1]
    
        if config.VERBOSE:
            print(f"audio_file: {audio_file_only}")

        # Extract features
        extracted_mfcc, signal, AUDIO_SAMPLE_RATE = utils.extract_mfcc(
            config.INPUT_FOLDER_PATH + audio_file_only,
            AUDIO_SAMPLE_RATE,
            N_FFT=best_audio_hyperparameters['N_FFT'],
            NUM_MFCC=best_audio_hyperparameters['NUM_MFCC'],
            HOP_LENGTH=best_audio_hyperparameters['HOP_LENGTH']
        )
    else:
        # Extract features
        extracted_mfcc, signal, AUDIO_SAMPLE_RATE = utils.extract_mfcc(
            audio_array,
            AUDIO_SAMPLE_RATE,
            N_FFT=best_audio_hyperparameters['N_FFT'],
            NUM_MFCC=best_audio_hyperparameters['NUM_MFCC'],
            HOP_LENGTH=best_audio_hyperparameters['HOP_LENGTH']
        )
    if config.VERBOSE:
        print(f"extracted_mfcc: {extracted_mfcc.shape}")

    # Reshape to make sure it fit pytorch model
    extracted_mfcc = np.repeat(extracted_mfcc[np.newaxis, np.newaxis, :, :], 3, axis=1)
    if config.VERBOSE:
        print(f"Reshaped extracted_mfcc: {extracted_mfcc.shape}")

    # Convert to tensor
    extracted_mfcc = torch.from_numpy(extracted_mfcc).float().to(config.device)
    
    if config.TRANSCRIBE:
        # string_audio = transcribe_audio.transcribe_audio(audio_array)
        string_audio = transcribe_audio.transcribe_audio(audio_array)
    # Load the model
    model = torch.load(config.AUDIO_MODEL_SAVE_PATH, map_location=torch.device('cpu'))

    model.to(config.device).eval()

    prediction = model(extracted_mfcc)
    prediction = torch.nn.functional.softmax(prediction, dim=1)
    prediction_numpy = prediction[0].cpu().detach().numpy()
    if config.VERBOSE:
        print(f"prediction: {prediction_numpy}")

    # Get the predicted label
    predicted_label = max(prediction_numpy)
    audio_emotion_index = prediction_numpy.tolist().index(predicted_label)
    emotion = config.EMOTION_INDEX[audio_emotion_index]
    if config.VERBOSE:
        print(f"Predicted emotion: {emotion} {round(predicted_label, 2)}")

    ret_string = utils.get_softmax_probs_string(prediction_numpy, list(config.EMOTION_INDEX.values()))
    if config.VERBOSE:
        print(f"\n\n\nPrediction labels:\n{ret_string}")
    
    return_objs = (emotion, string_audio, audio_emotion_index, prediction_numpy, extracted_mfcc)
    result_queue.put(item=return_objs)
    return None
