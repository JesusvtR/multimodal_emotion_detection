import time
import warnings
import numpy as np

import source.audio_analysis_utils.predict as audio_predict
import source.face_emotion_utils.predict as face_predict
import source.config as config
import source.audio_analysis_utils.transcribe_audio as transcribe_audio

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
        print(f"Resultado obtenido de {resultado_video}")
        result_queue.task_done()  # Marca el elemento como procesado
        resultado_audio = result_queue.get()
        print(f"Resultado obtenido de {resultado_audio}")
        result_queue.task_done()  # Marca el elemento como procesado
        fin = time.time()
        tiempo_transcurrido = fin - inicio
        print(f"Tiempo transcurrido: {tiempo_transcurrido} segundos")
        print("Grabación de audio y video completa.")
        emotion, emotion_index, tensor_video, image_array = resultado_video
        emotion, label, tensor_audio, extracted_mfcc = resultado_audio
        print("Tensores")
        
        # Convertir las listas a arrays numpy
        video_pred_array = np.array(tensor_video)
        audio_pred_array = np.array(tensor_audio)

        # Calcular el promedio de las predicciones
        combined_pred = (video_pred_array * 0.7 + audio_pred_array * 0.3)

        # Obtener el índice de la emoción con la probabilidad más alta
        combined_emotion_index = np.argmax(combined_pred)

        # Obtener la emoción correspondiente al índice
        combined_emotion = emotion_index_dict[combined_emotion_index]
        
        print(combined_emotion)
    
if __name__ == '__main__':
    main()
    