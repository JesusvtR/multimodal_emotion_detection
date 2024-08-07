import time
import warnings
import numpy as np

import source.audio_analysis_utils.predict as audio_predict
import source.face_emotion_utils.predict as face_predict
import source.config as config
import source.audio_analysis_utils.transcribe_audio as transcribe_audio
import ollama
from threading import Thread
from queue import Queue

# Ignorar todas las advertencias
warnings.filterwarnings("ignore")

def main():
    emotion_index_dict=config.EMOTION_INDEX
    transcribe_audio.init()
    while True:
        inicio = time.time()
        
        # Crear la cola para recibir los mensajes de los procesos
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

        # Obtiene resultados de la cola
        resultado_video = result_queue.get()
        if config.VERBOSE == True:
            print(f"Resultado obtenido de {resultado_video}")
        result_queue.task_done()  # Marca el elemento como procesado
        resultado_audio = result_queue.get()
        if config.VERBOSE == True:
            print(f"Resultado obtenido de {resultado_audio}")
        result_queue.task_done()  # Marca el elemento como procesado

        emotion_face, emotion_index, tensor_video, image_array, result_numpy = resultado_video
        emotion_speech, transcription, label, tensor_audio, extracted_mfcc = resultado_audio
        
        print(emotion_face)
        print(emotion_speech)
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
        
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        if config.VERBOSE == False:
            print(f"Tiempo transcurrido: {tiempo_transcurrido} segundos")
            #print("Grabación de audio y video completa.")
    
if __name__ == '__main__':
    main()
    