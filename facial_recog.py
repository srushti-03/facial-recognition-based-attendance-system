import os
import cv2
import face_recognition
import numpy as np


# tried cosine similarity but was giving many false positives
# tried processing every nth frame , but was not very accurate
# gives correct recognition upto 20-25 classes, proof of concept
# resolved the dlib face_recognition installation error


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_known_faces(self, dataset_path):
        """
        Load known faces and their embeddings from a dataset folder.

        Parameters:
        - dataset_path: Path to the dataset folder containing subfolders of person images.

        Returns:
        - None
        """
        for person_name in os.listdir(dataset_path):
            person_folder = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_folder):
                for filename in os.listdir(person_folder):
                    image_path = os.path.join(person_folder, filename)
                    if os.path.isfile(image_path):
                        image = cv2.imread(image_path)
                        if image is not None:
                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            encoding = face_recognition.face_encodings(image_rgb)
                            if len(encoding) > 0:
                                self.known_face_encodings.append(encoding[0])
                                self.known_face_names.append(person_name)

    def detect_known_faces(self, frame):
        """
        Detect known faces in a given frame.

        Parameters:
        - frame: Image frame (numpy array) captured from a video stream.

        Returns:
        - face_locations: List of face locations (bounding boxes) in the frame.
        - face_names: List of names corresponding to each detected face.
        - face_confidences: List of confidence levels (similarity scores) for each detected face.
        """
        face_locations = face_recognition.face_locations(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        face_names = []
        face_confidences = []
        for encoding in frame_encodings:
            # Compare current face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
            name = "Unknown"
            confidence = 0.0

            # Calculate face distances to get similarity scores
            face_distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = 1 - face_distances[best_match_index]

            face_names.append(name)
            face_confidences.append(confidence)

        return face_locations, face_names, face_confidences

# Initialize SimpleFacerec
sfr = SimpleFacerec()

# Load known faces and embeddings
dataset_path = r"C:\Users\LENOVO\Desktop\ig drones\face-recognition\dataset"
sfr.load_known_faces(dataset_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Detect Faces and Recognize
    face_locations, face_names, face_confidences = sfr.detect_known_faces(frame)

    # Display results
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_confidences):
        # Set the color based on whether the face is recognized or not
        if name == "Unknown":
            color = (0, 0, 255)  # Red for unknown
        else:
            color = (0, 255, 0)  # Green for known

        # Draw rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Display the name and confidence
        label = f"{name} ({confidence:.2f})"
        cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
