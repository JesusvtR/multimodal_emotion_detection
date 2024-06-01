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
best_audio_hyperparameters = utils.load_dict_from_json(config.AUDIO_BEST_HP_JSON_SAVE_PATH)
# if verbose:
#     print(f"Best hyperparameters, {best_hyperparameters}")

# Parámetros de grabación
duracion = 4  # Duración en segundos
frecuencia_muestreo = 44100  # Frecuencia de muestreo en Hz

def predict(result_queue, model_save_path=config.AUDIO_MODEL_SAVE_PATH, verbose=False, transcribe=True):
    
    # Grabación
    audio = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=2, dtype='int16')
    sd.wait()  # Espera hasta que la grabación se complete

    # Guardar en un archivo WAV
    wav.write(config.INPUT_FOLDER_PATH + 'output.wav', frecuencia_muestreo, audio)

    audio_file = utils.find_filename_match('output', config.INPUT_FOLDER_PATH)
    audio_file_only = audio_file.split(config.INPUT_FOLDER_PATH)[1]
  
    if verbose:
        print(f"audio_file: {audio_file_only}")

    # Clean the audio file
    # data.clean_single(audio_file, save_path=config.OUTPUT_FOLDER_PATH + audio_file_only.replace('.wav', '_clean.wav'), print_flag=True)
    # print("audio_file_cleaned")

    # Extract features
    extracted_mfcc, signal, sample_rate = utils.extract_mfcc(
        config.INPUT_FOLDER_PATH + audio_file_only,
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
        #transcribe_audio.transcribe_audio(file_path='C:\\git\\Video-Audio-Face-Emotion-Recognition\\input_files\\tu-eres-mi-hermano-del-alma-realmente-el-amigo.mp3')
        string_audio = transcribe_audio.transcribe_audio(audio=signal)
        emotion_transcribe = transcribe_audio.get_text_sentiment(string_audio)
        print(emotion_transcribe)

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
    
    return_objs = (emotion, predicted_label, prediction_numpy, extracted_mfcc)
    result_queue.put(item=return_objs)
    return None
