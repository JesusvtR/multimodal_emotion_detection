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
best_audio_hyperparameters = utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
# if verbose:
#     print(f"Best hyperparameters, {best_hyperparameters}")

# Parámetros de grabación
duration = 60  # Duración en segundos
sample_rate = 44100  # Frecuencia de muestreo en Hz

def predict(result_queue, model_save_path=config.AUDIO_MODEL_SAVE_PATH, verbose=config.VERBOSE, transcribe=config.TRANSCRIBE):
    
    # Parámetros de grabación
    duration = 60  # Duración en segundos
    sample_rate = 44100  # Frecuencia de muestreo en Hz
    save_audio = False
    # Record
    recorder = sr.Recognizer()
    source = sr.Microphone(sample_rate=16000)
    with source:
        recorder.adjust_for_ambient_noise(source)
        audio_data = recorder.listen(source, timeout=duration)
    
    wav_bytes = audio_data.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    
    if save_audio:
        # Guardar en un archivo WAV
        wav.write(config.INPUT_FOLDER_PATH + 'output.wav', sample_rate, audio_data)

        audio_file = utils.find_filename_match('output', config.INPUT_FOLDER_PATH)
        audio_file_only = audio_file.split(config.INPUT_FOLDER_PATH)[1]
    
        if verbose:
            print(f"audio_file: {audio_file_only}")

        # Extract features
        extracted_mfcc, signal, sample_rate = utils.extract_mfcc(
            config.INPUT_FOLDER_PATH + audio_file_only,
            sample_rate,
            N_FFT=best_audio_hyperparameters['N_FFT'],
            NUM_MFCC=best_audio_hyperparameters['NUM_MFCC'],
            HOP_LENGTH=best_audio_hyperparameters['HOP_LENGTH']
        )
    else:
        # Extract features
        extracted_mfcc, signal, sample_rate = utils.extract_mfcc(
            audio_array,
            sample_rate,
            N_FFT=best_audio_hyperparameters['N_FFT'],
            NUM_MFCC=best_audio_hyperparameters['NUM_MFCC'],
            HOP_LENGTH=best_audio_hyperparameters['HOP_LENGTH']
        )
    if verbose:
        print(f"extracted_mfcc: {extracted_mfcc.shape}")

    # Reshape to make sure it fit pytorch model
    extracted_mfcc = np.repeat(extracted_mfcc[np.newaxis, np.newaxis, :, :], 3, axis=1)
    if verbose:
        print(f"Reshaped extracted_mfcc: {extracted_mfcc.shape}")

    # Convert to tensor
    extracted_mfcc = torch.from_numpy(extracted_mfcc).float().to(config.device)
    
    if transcribe:
        string_audio = transcribe_audio.transcribe_audio(audio_array)
        #emotion_transcribe = transcribe_audio.get_text_sentiment(string_audio)
        if verbose:
            #print(emotion_transcribe)
            print(string_audio)

    # Load the model
    model = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.to(config.device).eval()

    prediction = model(extracted_mfcc)
    prediction = torch.nn.functional.softmax(prediction, dim=1)
    prediction_numpy = prediction[0].cpu().detach().numpy()
    if verbose:
        print(f"prediction: {prediction_numpy}")

    # Get the predicted label
    predicted_label = max(prediction_numpy)
    emotion = config.EMOTION_INDEX[prediction_numpy.tolist().index(predicted_label)]
    if verbose:
        print(f"Predicted emotion: {emotion} {round(predicted_label, 2)}")

    ret_string = utils.get_softmax_probs_string(prediction_numpy, list(config.EMOTION_INDEX.values()))
    if verbose:
        print(f"\n\n\nPrediction labels:\n{ret_string}")
    
    return_objs = (emotion, string_audio, predicted_label, prediction_numpy, extracted_mfcc)
    result_queue.put(item=return_objs)
    return None
