import cv2
import face_recognition
import numpy as np
import mongoengine as me
from flask import Flask, jsonify
import os
from datetime import datetime

# MongoDB Connection
MONGO_URI = "mongodb+srv://bossutkarsh30:YOCczedaElKny6Dd@cluster0.gixba.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
me.connect(db="alzheimers_db", host=MONGO_URI)

# Define MongoDB Model
class KnownPerson(me.Document):
    name = me.StringField(required=True)
    known_person_id = me.StringField(required=True, unique=True)
    patient_id = me.StringField(required=True)
    image_path = me.StringField(required=True)
    face_encoding = me.ListField(me.FloatField(), required=True)
    meta = {"collection": "known_person"} # Add this field to store image path

# Initialize Flask App
app = Flask(__name__)

# Live Face Recognition
def live_face_recognition(person_name):
    cap = cv2.VideoCapture(0)
    
    # Create images directory if it doesn't exist
    image_dir = os.path.join(os.path.dirname(__file__), 'face_images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    person_count = 0
    MAX_FACES = 45  # Set limit for training images
    face_encodings_list = []  # Store all encodings temporarily

    while True:
        if person_count >= MAX_FACES:
            print(f"Reached maximum number of faces ({MAX_FACES})")
            break

        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)

        for (top, right, bottom, left) in face_locations:
            if person_count >= MAX_FACES:
                break
                
            person_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            try:
                # Extract and save face image
                face_image = frame[top:bottom, left:right]
                image_filename = f"{person_name}_{timestamp}.jpg"
                image_path = os.path.join(image_dir, image_filename)
                cv2.imwrite(image_path, face_image)
                
                # Get face encoding
                face_encoding = face_recognition.face_encodings(rgb_frame, [face_locations[0]])[0]
                face_encodings_list.append(face_encoding)
                
                # Display remaining faces count
                remaining = MAX_FACES - person_count
                print(f"Captured face {person_count}/{MAX_FACES} for {person_name} ({remaining} remaining)")
                print(f"Saved image to: {image_path}")
                
            except IndexError:
                print(f"Could not encode face, skipping")
                person_count -= 1
                continue

            # Draw rectangle and label face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"{person_name} ({remaining} left)", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Return collected encodings for training
    return face_encodings_list, person_count

def train_face_recognition_model(person_name, face_encodings_list):
    """Train face recognition model and save one entry per person."""
    print("Starting model training...")
    
    if not face_encodings_list:
        print("No face encodings found")
        return False
    
    # Calculate average encoding from all captured faces
    average_encoding = np.mean(face_encodings_list, axis=0)
    
    # Delete existing entries for this person
    KnownPerson.objects(name=person_name).delete()
    
    # Generate timestamp and image path
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_path = os.path.join(
        os.path.dirname(__file__), 
        'face_images', 
        f"{person_name}_{timestamp}_average.jpg"
    )
    
    # Create single entry with average encoding
    try:
        new_person = KnownPerson(
            name=person_name,
            known_person_id=f"{person_name}_{timestamp}",
            patient_id=f"patient_{person_name}",
            image_path=image_path,  # Add the required image_path
            face_encoding=average_encoding.tolist()
        )
        new_person.save()
        print(f"Trained and saved model for: {person_name}")
        return True
    except Exception as e:
        print(f"Error saving face encoding: {str(e)}")  # Print the actual error
        return False

def identify_person(face_encoding):
    """Identify a person using their face encoding with distance threshold."""
    known_persons = KnownPerson.objects()
    
    if not known_persons:
        return "Unknown"
        
    known_encodings = [np.array(person.face_encoding) for person in known_persons]
    known_names = [person.name for person in known_persons]
    
    # Calculate face distances
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    
    # Find best match if any face distance is below threshold
    best_match_index = np.argmin(face_distances)
    if face_distances[best_match_index] < 0.6:  # Adjust threshold (0.6 is default)
        confidence = round((1 - face_distances[best_match_index]) * 100, 2)
        return f"{known_names[best_match_index]} ({confidence}%)"
    
    return "Unknown"

@app.route("/start_recognition/<person_name>", methods=["GET"])
def start_recognition(person_name):
    """Capture faces and train model."""
    print(f"Starting face capture for {person_name}...")
    face_encodings_list, total_faces = live_face_recognition(person_name)
    
    if face_encodings_list:
        success = train_face_recognition_model(person_name, face_encodings_list)
        return jsonify({
            "message": "Face capture and training completed" if success else "Training failed",
            "faces_captured": total_faces,
            "person_name": person_name,
            "success": success
        }), 200 if success else 500
    
    return jsonify({
        "message": "No faces captured",
        "success": False
    }), 400

@app.route("/train_model", methods=["GET"])
def train_model():
    """Train the face recognition model using saved faces."""
    success = train_face_recognition_model()
    return jsonify({
        "message": "Model training completed" if success else "Training failed",
        "success": success
    }), 200 if success else 500

# Add this new endpoint after the existing routes

@app.route("/test_identification", methods=["GET"])
def test_identification():
    """Test face identification using webcam."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        return jsonify({"error": "Could not access webcam"}), 500

    print("Starting face identification test...")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize frame for faster face detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Scale back face locations
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4) 
                         for (top, right, bottom, left) in face_locations]

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Identify the person
            name = identify_person(face_encoding)

            # Draw rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw filled rectangle for name background
            cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
            
            # Put name text
            cv2.putText(frame, name, (left + 6, top - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        # Display the number of faces detected
        cv2.putText(frame, f'Faces detected: {len(face_locations)}', (10, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1)

        cv2.imshow("Face Identification Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Identification test completed"}), 200

@app.route("/list_trained_faces", methods=["GET"])
def list_trained_faces():
    """List all trained faces in the database."""
    known_persons = KnownPerson.objects()
    trained_faces = [{
        "name": person.name,
        "id": person.known_person_id,
        "patient_id": person.patient_id
    } for person in known_persons]
    
    return jsonify({
        "total_faces": len(trained_faces),
        "trained_faces": trained_faces
    }), 200

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, port=5000)
