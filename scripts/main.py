#!/usr/bin/env python

import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from source import config as config
if config.ROS_SETUP == True:
    import rospy
    from multimodal_emotion_detection import EmotionData    

if config.PROMPT_LLM == True:
    from source.llm_utils import assistant
from source.audio_analysis_utils import predict as audio_predict
from source.face_emotion_utils import predict as face_predict
from source.audio_analysis_utils import transcribe_audio

from threading import Thread
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


def main():
    # Init audio
    transcribe_audio.init()
    #pub = rospy.Publisher('emotion_data', EmotionData, queue_size=10)
    if config.ROS_SETUP == True:
        loop = rospy.is_shutdown()
    else:
        loop = False
    while not loop:

        inicio = time.time()
        
        # Create queue
        result_queue = Queue()
        
        # Create processes
        p1 = Thread(target=face_predict.predict(result_queue),) 
        p2 = Thread(target=audio_predict.predict(result_queue),)

        # Processes list
        processes = [p1, p2]

        # Init processes
        for p in processes:
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Get facial result queue
        facial_prediction = result_queue.get()
        if config.VERBOSE == True:
            print(f"Facial prediction: {facial_prediction}")
        result_queue.task_done() 
        
        
        # Get speech result queue
        speech_prediction = result_queue.get()
        if config.VERBOSE == True:
            print(f"Audio prediction: {speech_prediction}")
        result_queue.task_done()  
        emotion_face, face_emotion_index, tensor_video = facial_prediction
        emotion_speech, transcription, speech_emotion_index, tensor_audio, extracted_mfcc = speech_prediction
        # Print emotions
        if config.VERBOSE == True:
            print(f'Face emotion: {emotion_face}')
            print(f'Speech emotion: {emotion_speech}')
        
        # Sent transcription prompt to LLM
        if config.PROMPT_LLM == True: 
            if(transcription == ""):
                print('No transcription')
            else:
                if config.VERBOSE == True:
                    print(transcription)
                assistant.ask(transcription, face_emotion_index, speech_emotion_index)
        
        # Time counter
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        if config.VERBOSE == True:
            print(f"Elapsed time: {tiempo_transcurrido} seconds")

        # Publish outputs
        if config.ROS_SETUP == True:
            emotion_data = EmotionData()
            emotion_data.emotion_face = face_emotion_index
            emotion_data.emotion_speech = speech_emotion_index
            emotion_data.transcription = transcription
            pub.publish(emotion_data)
    
if __name__ == '__main__':
    if config.ROS_SETUP == True:
        rospy.init_node('multimodal_emotion_detection')
    main()
    if config.ROS_SETUP == True:
        rospy.spin()
    