from src.SER.emotional_mesh import EmotionalMesh
from src.SER.emotion_detection import EmotionDetection
from speech_emotion_detection import recordAudio
from facial_emotion_detection import record_video
from threading import Thread, Event
import cv2
import time
import argparse

def main():
    # Start video stream
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    # Init EmotionDetection
    emotion_detection = EmotionDetection()

    # Stop event variable
    stop_event = Event()

    # Create threads
    t1 = Thread(target=record_video, args=(stop_event, emotion_detection, vs))
    t2 = Thread(target=recordAudio, args=(stop_event,))

    # Threads list
    tsk = [t1, t2]

    # Init threads
    for t in tsk:
        t.start()

    try:
        while True:
            # Main loop in active wait
            pass
    except KeyboardInterrupt:
        print("Stopping threads...")
        stop_event.set()  # Set flag
        
    # Wait for all threads to finish
    for t in tsk:
        t.join()
    print("Threads have stopped.")

    cv2.destroyAllWindows()
    vs.release()

if __name__ == '__main__':
    main()