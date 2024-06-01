import torch
import os
import csv
import librosa
import cv2
import time
import sys
import pyaudio
import wave
import numpy as np
import torch.nn.functional as F

from matplotlib import pyplot as plt
from PIL import Image as ImagePIL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import source.audio_analysis_utils.utils as audio_utils
import source.audio_analysis_utils.preprocess_data as audio_data
import source.audio_analysis_utils.predict as audio_predict
import source.audio_analysis_utils.transcribe_audio as transcribe_audio

import source.face_emotion_utils.utils as face_utils
import source.face_emotion_utils.face_mesh as face_mesh
import source.face_emotion_utils.face_config as face_config
import source.face_emotion_utils.predict as face_predict

import source.audio_face_combined.preprocess_main as preprocess_main
import source.pytorch_utils.visualize as pt_vis

import source.config as config


from multiprocessing import Process, Event
RECORD_SECONDS = 4

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 4
get_emotions = False
filename = "Record.wav"

def predict_video(
        image=None,
        video_mode=False,
        webcam_mode=False,
        model_save_path=config.FACE_MODEL_SAVE_PATH,
        best_hp_json_path=config.FACE_BEST_HP_JSON_SAVE_PATH,
        verbose=face_config.PREDICT_VERBOSE,
        imshow=face_config.SHOW_PRED_IMAGE,
        grad_cam=face_config.GRAD_CAM,
        grad_cam_on_video=face_config.GRAD_CAM_ON_VIDEO,
):
    """
    Predicts the emotion of the face in the image or video.
    Takes the full image, crops the face, detects the landmarks, and then runs the model on the face image and the landmarks.

    Parameters
    ----------
    image - path to image or video, or a numpy array of the image. Numpy array will only work if not video_mode or webcam_mode
        (Note: program currently only supports one face per image, if you'd like to add support for multiple faces, please submit a pull request.
        You'd just need to detect the faces using face_mesh.py or something similar, and then run the model on each face)
    video_mode - if True, will run the model on each frame of the video
    webcam_mode - if True, will run the model on each frame of the webcam
    model_save_path - path to the model to load
    imshow - if True, will show the image with the prediction
    verbose - if True, will print out the prediction probabilities

    Returns
    -------
    You'll get a tuple of the following based on the argments you pass in:

    if not webcam_mode and not video_mode:
        if grad_cam:
            Emotion name, emotion index, list of prediction probabilities, image as numpy, grad cam overlay as numpy
        else:
            Emotion name, emotion index, list of prediction probabilities, image as numpy
    else:
        None

    """

    best_hyperparameters = face_utils.load_dict_from_json(best_hp_json_path)
    if verbose:
        print(f"Best hyperparameters, {best_hyperparameters}")

    model = torch.load(model_save_path, map_location=torch.device('cpu'))
    model.to(config.device).eval()
    #model_text = transcribe_audio.init()
    cap = cv2.VideoCapture(0)
    # Inicializar la captura de audio
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK)
    while True:
        frames = []
        audio_frames = []
        sentiment = []

        init_time = time.time()
        
        while (time.time() - init_time) < RECORD_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_frames.append(audio_data)
            
        audio_data = b''.join(audio_frames)
        wf = wave.open(os.path.join(config.INPUT_FOLDER_PATH + filename), 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data)
        wf.close()
        
        if get_emotions == True:
            face_predict._get_prediction(best_hp=best_hyperparameters,
                img=frame,
                model=model,
                imshow=False,
                video_mode=True,
                verbose=verbose,
                grad_cam=True,
                grad_cam_on_video=grad_cam_on_video,
                feature_maps_flag=False)
            audio_predict.predict(filename)



def main():
    # Stop event variable
    stop_event = Event()

    # Create processes
    p1 = Process(target=record_video,args=(stop_event,)) 
    #p2 = Process(target=recordAudio,args=(stop_event,))

    # Processes list
    #processes = [p1, p2]
    processes = [p1]

    # Init processes
    for p in processes:
        p.start()

    try:
        while True:
            # Main loop in active wait
            pass
    except KeyboardInterrupt:
        print("Stopping processes...")
        
        stop_event.set()  # Set flag
        
    # Wait for all processes to finish
    for p in processes:
        p.join()
    print("Processes have stopped.")

if __name__ == "__main__":
    main()
