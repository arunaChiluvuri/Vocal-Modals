from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import threading

app = Flask(_name_, template_folder='templates')

# Initialize Mediapipe Hand Detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the Gesture Recognition Model (Update with correct model path)
model = tf.keras.models.load_model('mp_hand_gesture')

# Load class names (Update with correct file path)
with open('gesture.names', 'r') as f:
    classNames = f.read().split('\n')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# This is the function that will run the infinite loop
def process_video():
    while True:
        success, frame = cap.read()
        if not success:
            break

        x, y, _ = frame.shape
        frame = cv2.flip(frame, 1)  # Flip for natural movement
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for hand tracking
        result = hands.process(framergb)
        className = ""

        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])

                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict the gesture
                prediction = model.predict([landmarks])
                classID = np.argmax(prediction)
                className = classNames[classID] if classID < len(classNames) else "Unknown"

        # Display the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the frame in a window (optional)
        cv2.imshow("Gesture Recognition", frame)
        cv2.waitKey(1)

# Route for rendering the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video feed (for streaming or web page display)
def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route for streaming video frames
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the video processing in a separate thread
@app.before_first_request
def start_video_processing():
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True  # This ensures the thread will stop when the main app stops
    video_thread.start()

if _name_ == '_main_':
    app.run(debug=True)