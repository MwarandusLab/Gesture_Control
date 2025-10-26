import face_recognition

print("[INFO] Loading sample image...")
image = face_recognition.load_image_file("known_faces/Kheri.jpeg")  # fixed path
encodings = face_recognition.face_encodings(image)

if len(encodings) > 0:
    print("[SUCCESS] Face encoding created successfully!")
else:
    print("[FAIL] No face found in the image.")
