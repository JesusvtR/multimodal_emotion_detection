#!/usr/bin/env python

import sys
import os
import time
import ollama
import rospy

from source.audio_analysis_utils import predict as audio_predict
from source.face_emotion_utils import predict as face_predict
from source import config as config
from source.audio_analysis_utils import transcribe_audio
from threading import Thread
from queue import Queue

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from multimodal_emotion_detection import EmotionData

def main():
    # Init audio
    transcribe_audio.init()
    #pub = rospy.Publisher('emotion_data', EmotionData, queue_size=10)
    while not rospy.is_shutdown():

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

        # Get variables
        emotion_face, emotion_index, tensor_video, image_array, result_numpy = facial_prediction
        emotion_speech, transcription, label, tensor_audio, extracted_mfcc = speech_prediction
        
        # Print emotions
        print(emotion_face)
        print(emotion_speech)
        
        # Sent transcription prompt to LLM
        if config.PROMPT_LLM == True:
            # 
            if(transcription == ''):
                print('No transcription')
            else:
                print(transcription)
                response = ollama.chat(model='llama3.1', messages=[
                {
                  'role': 'user',
                  'content': transcription,
                },])
                print(response['message']['content'])
        
        # Time counter
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        if config.VERBOSE == False:
            print(f"Elapsed time: {tiempo_transcurrido} seconds")

        # Publish outputs
        emotion_data = EmotionData()
        emotion_data.emotion_face = emotion_face
        emotion_data.emotion_speech = emotion_speech
        emotion_data.transcription = transcription
        pub.publish(emotion_data)
    
if __name__ == '__main__':
    rospy.init_node('multimodal_emotion_detection')
    main()
    rospy.spin()
    