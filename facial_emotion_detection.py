
from src.SER.emotion_detection import EmotionDetection
import cv2
import time
import argparse

# Variables to calculate and show FPS
counter, fps = 0, 0
fps_avg_frame_count = 10
text_location = (20, 24)
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
start_time = time.time()

# Init time of program
init_time = time.time()

def draw_fps(image, fps):
    fps_text = 'FPS = {:.1f}'.format(fps)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    return image

def calculate_fps():
    global counter, fps, start_time
    counter += 1
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()
    return fps

def savefps(fps, max_time, fps_file):
    fps_file.write('{:.1f}, {:.1f}\n'.format(fps, (time.time() - init_time)))
    if (time.time() - init_time) >= max_time:
        print("FPS save is over")
        return False
    return True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savefps',
                    action='store_true',
                    default=False, required=False,
                    help='Activa el guardado de fps en fichero .csv')
    parser.add_argument('--time',
                    type=int,
                    default=30, required=False,
                    help='Duracion en segundos del guardado de datos (int). Default: 30s')
    return parser.parse_args()

def draw_predict(pred, image):
    if pred == 0:
        emotion = "Neutral"
    if pred == 1:
        emotion = "Anger"
    elif pred == 2:
        emotion = "Contempt"
    elif pred == 3:
        emotion = "Disgust"
    elif pred == 4:
        emotion = "Fear"
    elif pred == 5:
        emotion = "Happy"
    elif pred == 6:
        emotion = "Sadness"
    else:
        emotion = "Surprise"
    cv2.putText(image, emotion, (20, 300), cv2.FONT_HERSHEY_PLAIN,
                2, text_color, font_thickness)

def record_video(stop_event):
    # Start video stream
    vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    # Init EmotionDetection
    emotion_detection = EmotionDetection()
    while not stop_event.is_set():
        # Read image from stream
        ret,image = vs.read()
        #image = cv2.flip(image, 1)

        # Process image
        pred = emotion_detection.predict(image)
        draw_predict(pred, image)

        # Calculate the FPS
        fps = calculate_fps()

        # Draw the FPS
        image = draw_fps(image, fps)

        # Show image
        cv2.imshow("imagen", image)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('imagen', cv2.WND_PROP_VISIBLE) < 1):
            break
    cv2.destroyAllWindows()
    vs.release()
