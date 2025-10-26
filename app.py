"""
app.py
Face + finger gesture controlled appliance toggles prototype.
Requires: face_recognition==1.4.0, mediapipe, opencv-python, numpy, imutils, pyserial
"""

import os
import time
from collections import deque, defaultdict

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
import serial
import serial.tools.list_ports

# -------------------------
# Configuration
# -------------------------
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5
FRAME_RESIZE_WIDTH = 640
STABLE_FACE_FRAMES = 3
STABLE_GESTURE_FRAMES = 6
GESTURE_COOLDOWN_S = 1.5
SERIAL_BAUD_RATE = 9600
SERIAL_TIMEOUT = 1

# Finger count → appliance mapping
GESTURE_MAP = {
    1: "Light A",
    2: "Light B",
    3: "Fan",
    5: "ALL_OFF"  # Special gesture to turn off everything
}


# -------------------------
# Serial Communication
# -------------------------
def list_serial_ports():
    """List available serial ports."""
    ports = serial.tools.list_ports.comports()
    available = []
    for port in ports:
        available.append(port.device)
    return available


def connect_serial(port=None):
    """Attempt to connect to Arduino via serial."""
    try:
        if port is None:
            # Auto-detect Arduino
            ports = list_serial_ports()
            if not ports:
                print("[SERIAL] No serial ports found.")
                return None
            
            # Try to find Arduino (usually has 'USB' or 'ACM' in name)
            arduino_port = None
            for p in ports:
                if 'USB' in p.upper() or 'ACM' in p.upper() or 'COM' in p.upper():
                    arduino_port = p
                    break
            
            if arduino_port is None:
                arduino_port = ports[0]  # Fallback to first port
            
            port = arduino_port
        
        ser = serial.Serial(port, SERIAL_BAUD_RATE, timeout=SERIAL_TIMEOUT)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"[SERIAL] Connected to {port} at {SERIAL_BAUD_RATE} baud")
        return ser
        
    except Exception as e:
        print(f"[SERIAL] Connection failed: {e}")
        return None


def send_serial_data(ser, data):
    """Send data via serial if connection exists."""
    if ser and ser.is_open:
        try:
            ser.write(f"{data}\n".encode())
            print(f"[SERIAL] Sent: {data}")
            return True
        except Exception as e:
            print(f"[SERIAL] Send error: {e}")
            return False
    return False


# -------------------------
# Load known faces
# -------------------------
def load_known_faces(known_dir):
    known_encodings, known_names = [], []
    for fn in os.listdir(known_dir):
        path = os.path.join(known_dir, fn)
        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        # Load and convert image properly
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(f"[WARN] Could not read {fn}, skipping.")
            continue
        
        # Convert BGR to RGB (face_recognition uses RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            # Detect faces first
            face_locations = face_recognition.face_locations(img_rgb, model="hog")
            
            if len(face_locations) == 0:
                print(f"[WARN] No face found in {fn}, skipping.")
                continue
            
            # Encode the first face found
            encs = face_recognition.face_encodings(img_rgb, face_locations)
            
            if len(encs) == 0:
                print(f"[WARN] Could not encode face in {fn}, skipping.")
                continue
                
            known_encodings.append(encs[0])
            known_names.append(name)
            print(f"[INFO] Loaded {name} from {fn}")
            
        except Exception as e:
            print(f"[WARN] Failed encoding {fn}: {e}")
            continue

    return known_encodings, known_names


# -------------------------
# Mediapipe Hand Detection
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def count_fingers_mediapipe(hand_landmarks, handedness_label="Right"):
    """Return the number of fingers detected as raised."""
    if not hand_landmarks:
        return None

    lm = hand_landmarks.landmark
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    fingers = []

    # Thumb
    try:
        thumb_tip = lm[tips_ids[0]]
        thumb_ip = lm[pip_ids[0]]
        if handedness_label == "Right":
            thumb_is_open = thumb_tip.x < thumb_ip.x
        else:
            thumb_is_open = thumb_tip.x > thumb_ip.x
    except Exception:
        thumb_is_open = False

    fingers.append(1 if thumb_is_open else 0)

    # Other fingers
    for tip_id, pip_id in zip(tips_ids[1:], pip_ids[1:]):
        try:
            tip = lm[tip_id]
            pip = lm[pip_id]
            fingers.append(1 if tip.y < pip.y else 0)
        except Exception:
            fingers.append(0)

    return sum(fingers)


# -------------------------
# Main App
# -------------------------
def main():
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    if len(known_encodings) == 0:
        print("[ERROR] No known faces loaded. Add images into 'known_faces/' folder.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    face_window = deque(maxlen=STABLE_FACE_FRAMES)
    gesture_window = deque(maxlen=STABLE_GESTURE_FRAMES)

    last_action_time = defaultdict(lambda: 0.0)
    appliance_state = {name: False for name in GESTURE_MAP.values()}

    mp_h = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    # Initialize serial connection
    ser = connect_serial()
    
    print("[INFO] Webcam started. Press 'q' to quit, 'r' to reconnect serial.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Resize frame
        scale = FRAME_RESIZE_WIDTH / frame.shape[1]
        small = cv2.resize(frame, (FRAME_RESIZE_WIDTH, int(frame.shape[0] * scale)))
        
        # Convert to RGB for face_recognition (critical!)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Face detection & recognition (process every frame for better detection)
        recognized_name = None
        face_locations = []
        face_encodings = []
        
        try:
            # Detect faces using HOG (faster) - returns list of (top, right, bottom, left)
            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            
            if face_locations:
                # Encode faces - MUST pass face_locations
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                
                # Compare with known faces
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_encodings, 
                        face_encoding, 
                        tolerance=TOLERANCE
                    )
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_idx = np.argmin(face_distances)
                        if matches[best_idx]:
                            recognized_name = known_names[best_idx]
                            break
                            
        except Exception as e:
            if frame_count % 30 == 0:  # Log every 30 frames to avoid spam
                print(f"[WARN] Face processing error: {e}")

        # Update face stability window
        face_window.append(recognized_name)
        stable_name = None
        if len(face_window) == face_window.maxlen:
            names = [n for n in face_window if n]
            if names:
                stable_name = max(set(names), key=names.count)

        # Hand detection
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = mp_h.process(small_rgb)
        finger_count, handedness_label = None, "Right"

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            if results.multi_handedness:
                handedness_label = results.multi_handedness[0].classification[0].label
            finger_count = count_fingers_mediapipe(landmarks, handedness_label)
            mp_draw.draw_landmarks(small, landmarks, mp_hands.HAND_CONNECTIONS)

        # Update gesture stability window
        gesture_window.append(finger_count)
        stable_gesture = None
        if len(gesture_window) == gesture_window.maxlen:
            vals = [v for v in gesture_window if v is not None]
            if vals and all(v == vals[0] for v in vals):
                stable_gesture = vals[0]

        # Trigger appliance only if both face + gesture stable
        if stable_name and stable_gesture in GESTURE_MAP:
            appliance = GESTURE_MAP[stable_gesture]
            now = time.time()
            
            # Special case: 5 fingers turns off everything
            if stable_gesture == 5:
                if now - last_action_time["ALL_OFF"] > GESTURE_COOLDOWN_S:
                    # Turn off all appliances
                    for app_name in appliance_state.keys():
                        appliance_state[app_name] = False
                    print(f"[ACTION] {stable_name} turned OFF all appliances (5 fingers)")
                    
                    # Send to Arduino
                    send_serial_data(ser, "5")
                    
                    last_action_time["ALL_OFF"] = now
            else:
                # Normal toggle for individual appliances
                if now - last_action_time[appliance] > GESTURE_COOLDOWN_S:
                    appliance_state[appliance] = not appliance_state[appliance]
                    state_str = "ON" if appliance_state[appliance] else "OFF"
                    print(f"[ACTION] {stable_name} triggered {appliance} -> {state_str} (fingers: {stable_gesture})")
                    
                    # Send to Arduino (send the finger count)
                    send_serial_data(ser, str(stable_gesture))
                    
                    last_action_time[appliance] = now

        # Draw face rectangles
        for (top, right, bottom, left) in face_locations:
            color = (0, 255, 0) if recognized_name else (0, 0, 255)
            cv2.rectangle(small, (left, top), (right, bottom), color, 2)
            label = recognized_name if recognized_name else "Unknown"
            cv2.putText(small, label, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Overlay info
        face_label = stable_name if stable_name else "Unknown"
        finger_label = stable_gesture if stable_gesture is not None else (finger_count if finger_count else "-")
        serial_status = "Connected" if (ser and ser.is_open) else "Disconnected"
        overlay = f"Face: {face_label} | Fingers: {finger_label} | Serial: {serial_status}"
        cv2.putText(small, overlay, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        legend_y = 50
        for k, v in GESTURE_MAP.items():
            if v == "ALL_OFF":
                cv2.putText(small, f"{k} fingers -> Turn OFF All", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                st = "ON" if appliance_state[v] else "OFF"
                cv2.putText(small, f"{k} -> {v} [{st}]", (10, legend_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
            legend_y += 25

        cv2.imshow("Face + Gesture Control", small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r") or key == ord("R"):
            # Reconnect serial
            print("[SERIAL] Reconnecting...")
            if ser and ser.is_open:
                ser.close()
            ser = connect_serial()

    cap.release()
    cv2.destroyAllWindows()
    mp_h.close()
    
    # Close serial connection
    if ser and ser.is_open:
        ser.close()
        print("[SERIAL] Connection closed.")
    
    print("[INFO] Exiting.")


if __name__ == "__main__":
    main()