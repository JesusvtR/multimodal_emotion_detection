from speech_emotion_detection import recordAudio
from facial_emotion_detection import record_video
from multiprocessing import Process, Event

def main():
    # Stop event variable
    stop_event = Event()

    # Create processes
    p1 = Process(target=record_video,args=(stop_event,)) 
    p2 = Process(target=recordAudio,args=(stop_event,))

    # Processes list
    processes = [p1, p2]

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

if __name__ == '__main__':
    main()