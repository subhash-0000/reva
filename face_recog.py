import cv2
import face_recognition
import numpy as np
import os
import mongoengine as me
from flask import Flask

# Connect to MongoDB
me.connect("face_recognition_db", host="mongodb://localhost:27017/face_recognition_db")

# Define MongoDB Models
class KnownPerson(me.Document):
    known_person_id = me.StringField(required=True, unique=True)
    name = me.StringField(required=True)
    encoding = me.ListField(me.FloatField(), required=True)  # Store face encoding

# Initialize Flask App
app = Flask(__name__)

# Load known faces from MongoDB
def load_known_faces():
    known_face_encodings = []
    known_face_names = []
    
    for person in KnownPerson.objects():
        known_face_encodings.append(np.array(person.encoding))
        known_face_names.append(person.name)
    
    return known_face_encodings, known_face_names

# Start Live Face Recognition
def live_face_recognition():
    known_face_encodings, known_face_names = load_known_faces()

    cap = cv2.VideoCapture(0)  # Open Webcam
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Draw a rectangle around the face and label it
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/start_recognition", methods=["GET"])
def start_recognition():
    """Trigger face recognition when this route is accessed."""
    live_face_recognition()
    return "Face Recognition Started", 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
