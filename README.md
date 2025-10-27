# 🖐️ Face + Gesture Controlled Appliance System

A real-time face recognition and hand gesture control system that allows authorized users to control appliances (lights, fans) using finger gestures. The system sends commands via serial communication to an Arduino for physical appliance control.

## ✨ Features

- **👤 Face Recognition**: Only authorized users can control appliances
- **🖐️ Gesture Control**: Control different appliances with finger counts
  - 1 finger → Toggle Light A
  - 2 fingers → Toggle Light B
  - 3 fingers → Toggle Fan
  - 5 fingers → Emergency OFF (all appliances)
- **🔌 Arduino Integration**: Serial communication to control real hardware
- **🔄 Hot-Reload**: Press 'R' to reconnect Arduino without restarting
- **⚡ Non-Blocking**: Works even when Arduino is not connected
- **📊 Real-time Feedback**: On-screen status display and terminal logging

## 🎥 Demo

The system detects your face, counts your raised fingers, and sends commands to Arduino when both are stable (prevents accidental triggers).

## 📋 Prerequisites

### System Requirements
- **Python 3.11** (Critical for compatibility)
- Webcam
- Arduino Mega (or compatible board) - Optional
- Windows/Linux/MacOS

### Important Notes
- ⚠️ **Python 3.11 is required** - Newer versions have dependency conflicts
- ⚠️ **NumPy 1.23.5 specifically** - Version 2.0+ breaks compatibility
- ⚠️ **Use virtual environment** - Highly recommended to avoid conflicts

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/MwarandusLab/Gesture_Control.git
cd Gesture_Control
```

### Step 2: Create Virtual Environment

**IMPORTANT**: Always use a virtual environment to isolate dependencies!

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install dlib

dlib is required for face recognition and can be tricky to install. Use the pre-compiled version:

**Windows Users:**
```bash
# Download pre-compiled dlib wheel from:
# https://github.com/MwarandusLab/dlib

# Install the downloaded .whl file
pip install dlib-19.24.0-cp311-cp311-win_amd64.whl
```

**Linux/Mac Users:**
```bash
# Install dependencies first
sudo apt-get install cmake
sudo apt-get install libopenblas-dev liblapack-dev

# Then install dlib
pip install dlib
```

### Step 4: Install Dependencies

⚠️ **Critical**: Install NumPy first with the specific version:

```bash
# Install NumPy 1.23.5 first (REQUIRED - do not skip!)
pip install numpy==1.23.5

# Then install other dependencies
pip install face_recognition==1.4.0
pip install mediapipe
pip install opencv-python
pip install imutils
pip install pyserial
```

**Or use requirements.txt:**

```bash
pip install -r requirements.txt
```

### Step 5: Setup Known Faces

Create a `known_faces` folder in the project directory and add photos of authorized users:

```bash
mkdir known_faces
```

Add face images (one face per image):
- `known_faces/James.jpg`
- `known_faces/Kheri.jpeg`
- `known_faces/YourName.png`

**Image Guidelines:**
- ✅ Clear, front-facing photos
- ✅ Good lighting
- ✅ One face per image
- ✅ Minimum 300x300 pixels
- ✅ Supported formats: .jpg, .jpeg, .png

## 🎮 Usage

### Running the Application

```bash
python app.py
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Reconnect Arduino (hot-reload) |

### Gesture Controls

| Gesture | Action |
|---------|--------|
| 🖕 1 finger | Toggle Light A |
| ✌️ 2 fingers | Toggle Light B |
| 🤟 3 fingers | Toggle Fan |
| 🖐️ 5 fingers | Turn OFF everything |

## 🔌 Arduino Setup

### Hardware Connection
1. Connect Arduino Mega via USB cable
2. Note the COM port (e.g., COM3, /dev/ttyUSB0)
3. Upload the Arduino sketch

### Arduino Code Example

```cpp
void setup() {
  Serial.begin(9600);
  
  // Setup your relay pins
  pinMode(2, OUTPUT); // Light A
  pinMode(3, OUTPUT); // Light B
  pinMode(4, OUTPUT); // Fan
  
  // Initialize all OFF
  digitalWrite(2, LOW);
  digitalWrite(3, LOW);
  digitalWrite(4, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    switch(cmd) {
      case '1':
        // Toggle Light A
        digitalWrite(2, !digitalRead(2));
        Serial.println("Light A toggled");
        break;
        
      case '2':
        // Toggle Light B
        digitalWrite(3, !digitalRead(3));
        Serial.println("Light B toggled");
        break;
        
      case '3':
        // Toggle Fan
        digitalWrite(4, !digitalRead(4));
        Serial.println("Fan toggled");
        break;
        
      case '5':
        // Emergency OFF - all appliances
        digitalWrite(2, LOW);
        digitalWrite(3, LOW);
        digitalWrite(4, LOW);
        Serial.println("All OFF");
        break;
    }
  }
}
```

## 📦 Requirements File

Create `requirements.txt` with exact versions:

```txt
numpy==1.23.5
face_recognition==1.4.0
mediapipe
opencv-python
imutils
pyserial
```

## 🐛 Troubleshooting

### Common Issues

**1. "No module named 'dlib'"**
- Install dlib from: https://github.com/MwarandusLab/dlib
- Use pre-compiled wheel for Windows

**2. NumPy compatibility errors**
```bash
pip uninstall numpy
pip install numpy==1.23.5
```

**3. "No known faces loaded"**
- Ensure images are in `known_faces/` folder
- Check image formats (.jpg, .jpeg, .png)
- Verify face is clearly visible in images

**4. Face not recognized**
- Improve lighting conditions
- Face the camera directly
- Try increasing `TOLERANCE` in code (0.5 → 0.6)

**5. Serial connection fails**
- Check Arduino is connected via USB
- Verify correct COM port
- Press 'R' to reconnect
- Install Arduino drivers if needed

**6. Python version issues**
- Must use Python 3.11
- Newer versions (3.12+) have compatibility issues
- Check version: `python --version`

## ⚙️ Configuration

Edit `app.py` to customize:

```python
# Face recognition
TOLERANCE = 0.5  # Lower = stricter matching

# Gesture stability
STABLE_FACE_FRAMES = 3  # Frames for stable face
STABLE_GESTURE_FRAMES = 6  # Frames for stable gesture

# Cooldown
GESTURE_COOLDOWN_S = 1.5  # Seconds between actions

# Serial
SERIAL_BAUD_RATE = 9600  # Match Arduino baud rate
```

## 📁 Project Structure

```
Gesture_Control/
├── app.py                 # Main application
├── known_faces/           # Face images directory
│   ├── James.jpg
│   └── Kheri.jpeg
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (ignored in git)
```

## 🔒 Security Features

- Face authentication required before any gesture control
- Stability windows prevent accidental triggers
- Cooldown timers prevent rapid repeated actions
- Only authorized faces (in known_faces/) can control appliances

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**MwarandusLab**
- GitHub: [@MwarandusLab](https://github.com/MwarandusLab)

## 🙏 Acknowledgments

- Face recognition library by Adam Geitgey
- MediaPipe by Google for hand detection
- dlib by Davis King

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Ensure you're using Python 3.11 and NumPy 1.23.5

---

**⭐ If this project helped you, please give it a star!**
