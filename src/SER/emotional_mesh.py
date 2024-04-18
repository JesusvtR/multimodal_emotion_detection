import mediapipe as mp
import numpy as np
import cv2
import math

class EmotionalMesh:
    def __init__(self, static=False, refine=False):
        # Init FaceMesh variables
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static,
            max_num_faces=1,
            refine_landmarks=refine,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Results and return image
        self.results = None
        self.image = None

        # Indexes of 'Emotional Mesh' in  Mediapipe
        self.indexes = [61, 291, 0, 17, 50, 280, 48, 4, 278,
            206, 426, 133, 130, 159, 145, 362, 359, 386, 374, 122,
            351, 46, 105, 107, 276, 334, 336]

        # Coordinates of 'Emotional Mesh' - Tuple (x, y)
        self.coordinates = []

        # Relative coordinates of 'Emotional Mesh' - Tuple (x, y)
        self.rcoordinates = []

        # Angles
        # Format: [angle1, angle2, angle3, ...]
        self.num_angles = 21
        self.angles = np.zeros((1, self.num_angles))

    def reset_coordinates(self):
        self.coordinates.clear()

    def reset_rcoordinates(self):
        self.rcoordinates.clear()

    def reset_angles(self):
        for i in range(self.num_angles):
            self.angles[0][i] = 0

    def distance(self, point1, point2):
        x0 = point1[0]
        y0 = point1[1]
        x1 = point2[0]
        y1 = point2[1]
        return math.sqrt((x0 - x1)**2+(y0 - y1)**2)

    def angle(self, point1, point2, point3):
        side1 = self.distance(point2, point3)
        side2 = self.distance(point1, point3)
        side3 = self.distance(point1, point2)
        
        angle = math.degrees(math.acos((side1**2+side3**2-side2**2)/(2*side1*side3)))
        return angle

    def calculate_angles(self):
        index = 0

        # Angle 0
        self.angles[0][index] = self.angle(self.coordinates[7], self.coordinates[1], 
            self.coordinates[2])
        index += 1
        # Angle 1
        self.angles[0][index] = self.angle(self.coordinates[2], self.coordinates[1], 
            self.coordinates[3])
        index += 1
        # Angle 2
        self.angles[0][index] = self.angle(self.coordinates[0], self.coordinates[2], 
            self.coordinates[1])
        index += 1
        # Angle 3
        self.angles[0][index] = self.angle(self.coordinates[1], self.coordinates[7], 
            self.coordinates[8])
        index += 1
        # Angle 4
        self.angles[0][index] = self.angle(self.coordinates[0], self.coordinates[7], 
            self.coordinates[1])
        index += 1
        # Angle 5
        self.angles[0][index] = self.angle(self.coordinates[8], self.coordinates[5], 
            self.coordinates[1])
        index += 1
        # Angle 6
        self.angles[0][index] = self.angle(self.coordinates[8], self.coordinates[10], 
            self.coordinates[1])
        index += 1
        # Angle 7
        self.angles[0][index] = self.angle(self.coordinates[18], self.coordinates[5], 
            self.coordinates[8])
        index += 1
        # Angle 8
        self.angles[0][index] = self.angle(self.coordinates[8], self.coordinates[7], 
            self.coordinates[20])
        index += 1
        # Angle 9
        self.angles[0][index] = self.angle(self.coordinates[26], self.coordinates[7], 
            self.coordinates[23])
        index += 1
        # Angle 10
        self.angles[0][index] = self.angle(self.coordinates[7], self.coordinates[20], 
            self.coordinates[18])
        index += 1
        # Angle 11
        self.angles[0][index] = self.angle(self.coordinates[20], self.coordinates[18], 
            self.coordinates[5])
        index += 1
        # Angle 12
        self.angles[0][index] = self.angle(self.coordinates[17], self.coordinates[16], 
            self.coordinates[18])
        index += 1
        # Angle 13
        self.angles[0][index] = self.angle(self.coordinates[26], self.coordinates[25], 
            self.coordinates[24])
        index += 1
        # Angle 14
        self.angles[0][index] = self.angle(self.coordinates[20], self.coordinates[26], 
            self.coordinates[25])
        index += 1
        # Angle 15
        self.angles[0][index] = self.angle(self.coordinates[25], self.coordinates[24], 
            self.coordinates[16])
        index += 1
        # Angle 16
        self.angles[0][index] = self.angle(self.coordinates[24], self.coordinates[16], 
            self.coordinates[17])
        index += 1
        # Angle 17
        self.angles[0][index] = self.angle(self.coordinates[18], self.coordinates[20], 
            self.coordinates[26])
        index += 1
        # Angle 18
        self.angles[0][index] = self.angle(self.coordinates[5], self.coordinates[1], 
            self.coordinates[10])
        index += 1
        # Angle 19
        self.angles[0][index] = self.angle(self.coordinates[10], self.coordinates[1], 
            self.coordinates[7])
        index += 1
        # Angle 20
        self.angles[0][index] = self.angle(self.coordinates[10], self.coordinates[8], 
            self.coordinates[5])

    def update_coordinates(self, face_landmarks):
        height, width, _ = self.image.shape
        for index in self.indexes:
            x = face_landmarks.landmark[index].x
            y = face_landmarks.landmark[index].y
            self.coordinates.append((x, y))
            self.rcoordinates.append((int(x*width), int(y*height)))
    
    def process(self):
        self.results = self.face_mesh.process(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        if self.results.multi_face_landmarks:
            for face_landmarks in self.results.multi_face_landmarks:
                self.reset_coordinates()
                self.reset_rcoordinates()
                self.reset_angles()
                self.update_coordinates(face_landmarks)
                self.calculate_angles()

    def draw_emotional_mesh(self):
        for coord in self.rcoordinates:
            cv2.circle(self.image, (coord[0], coord[1]), 2, (0, 255, 0), -1)

    def set_image(self, image):
        self.image = image.copy()

    def get_image(self):
        return self.image

    def get_angles(self):
        angles = np.copy(self.angles)
        return angles

    def face_mesh_detected(self):
        if self.results.multi_face_landmarks:
            return True
        return False 