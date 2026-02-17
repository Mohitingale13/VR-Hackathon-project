import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import time
from threading import Thread

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hackathon_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

current_stage = "up"
rep_count = 0
error_counter = 0

# --- ROUTES ---

@app.route('/')
def home():
    # This serves the NEW Dashboard page
    return render_template('home.html')

@app.route('/vr')
def vr_session():
    # Get parameters from URL (sent by the wizard)
    env = request.args.get('env', 'gym') 
    exercise = request.args.get('ex', 'pushup')
    
    # You can pass 'exercise' to the template if you want to change logic based on it
    return render_template('index.html', env=env, exercise=exercise)
@app.route('/about')
def about():
    return render_template('about.html')

# --- AI & WEBCAM LOGIC ---
# (Keep your existing math logic functions here: calculate_angle, etc.)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def process_webcam():
    global current_stage, rep_count
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # ... (Keep your existing MediaPipe logic exactly as is) ...
        # For brevity, I am assuming the logic remains identical to your code
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            
            # (Use your existing Pushup Logic Here)
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle = calculate_angle(shoulder, elbow, wrist)
            
            if angle < 90: current_stage = "down"
            if angle > 160 and current_stage == 'down':
                current_stage = "up"
                rep_count += 1
                socketio.emit('play_audio', {'file': 'good'})

            socketio.emit('update_vr', {'reps': rep_count, 'status': "Push!" if current_stage == "up" else "Up!"})

        # cv2.imshow('AI Vision', image) # Optional: Comment out if you don't want the popup on laptop
        if cv2.waitKey(10) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

# --- AI COACH CHAT LOGIC ---
@socketio.on('send_chat')
def handle_chat(data):
    # (Keep your existing mock AI logic)
    text = data['message'].lower()
    response = "I can help with workouts! Ask away."
    
    if 'diet' in text: response = "Eat lean protein 2 hours before working out."
    elif 'stress' in text: response = "Try 4-7-8 breathing to relax."
    elif 'exercise' in text: response = "Start with 3 sets of 12 reps."
    
    emit('receive_chat', {'role': 'ai', 'text': response})

if __name__ == '__main__':
    t = Thread(target=process_webcam)
    t.daemon = True
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)