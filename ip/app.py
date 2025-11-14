import cv2
import csv
from datetime import datetime, date, time
# We no longer need the separate 'import time'
import mediapipe as mp
import math
from scipy.spatial import distance as dist
import numpy as np
import os
import json
import pandas as pd
import plotly
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, Response, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import pytz
import face_recognition
import pickle
import atexit
from queue import Queue
from threading import Thread

# --- Constants & Calibration ---
FOCUS_PITCH_THRESHOLD = 25
FOCUS_YAW_THRESHOLD = 30
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
HAND_DETECTION_SKIP_FRAMES = 10
SCORE_WEIGHTS = {
    "Attentive": 1.0,
    "Looking Down": 0.6,
    "Looking Away": 0.5,
    "Face Covered": 0.3,
    "Sleepy": 0.0
}
CAMERA_INDEX = 0

# --- MediaPipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- Eye Landmark Indices ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# --- 3D Head Pose Model Points ---
model_points_3d = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

# --- Load Face Recognition Encodings ---
print("[INFO] Loading face encodings...")
try:
    with open("encodings.pkl", "rb") as f:
        known_face_data = pickle.load(f)
    print("[INFO] Encodings loaded successfully.")
except FileNotFoundError:
    print("[ERROR] 'encodings.pkl' not found. Please run 'encode_faces.py' first.")
    known_face_data = {"encodings": [], "names": []}
# ---

# --- Flask App Setup ---
app = Flask(__name__)

# --- Database Path Setup ---
basedir = os.path.abspath(os.path.dirname(__file__))
instance_folder = os.path.join(basedir, 'instance')
os.makedirs(instance_folder, exist_ok=True)
db_path = os.path.join(instance_folder, 'app.db')

app.config['SECRET_KEY'] = 'your_very_secret_key_here_123'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False 

# --- Flask-Mail Configuration ---
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # <-- REPLACE THIS
app.config['MAIL_PASSWORD'] = 'your-16-char-app-password' # <-- REPLACE THIS
mail = Mail(app)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'


# --- Background Database Writer ---
db_queue = Queue()

def db_writer():
    while True:
        log_data = db_queue.get() 
        if log_data is None: 
            break
        
        with app.app_context():
            try:
                log_entry = AttentivenessLog(
                    roll_no=log_data['roll_no'],
                    status=log_data['status'],
                    attentiveness_score=log_data['attentiveness_score']
                )
                db.session.add(log_entry)
                db.session.commit()
            except Exception as e:
                print(f"DB Write Error: {e}")
                db.session.rollback()
        
        db_queue.task_done() 

db_thread = Thread(target=db_writer, daemon=True)
db_thread.start()

def stop_db_thread():
    print("Stopping DB worker thread...")
    db_queue.put(None) 
    db_thread.join()
    print("DB worker stopped.")

atexit.register(stop_db_thread)
# --- END of Background DB Writer ---


# --- Database Models ---

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Timetable(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hour = db.Column(db.Integer, unique=True, nullable=False) 
    subject = db.Column(db.String(100), default='Not Set')
    teacher_name = db.Column(db.String(100), default='Not Set')
    teacher_email = db.Column(db.String(150), default='Not Set')

class AttentivenessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    roll_no = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    attentiveness_score = db.Column(db.Float, nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- CV Helper Functions ---
def calculate_ear(eye_landmarks, face_landmarks, frame_shape):
    h, w = frame_shape
    coords = []
    for idx in eye_landmarks:
        lm = face_landmarks.landmark[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    A = math.dist(coords[1], coords[5])
    B = math.dist(coords[2], coords[4])
    C = math.dist(coords[0], coords[3])
    if C == 0: return 0.3
    ear = (A + B) / (2.0 * C)
    return ear

def check_overlap(bbox1, bbox2):
    if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]: return False
    if bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]: return False
    return True

def get_bbox_and_centroid(landmarks, h, w):
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        x_max = max(x_max, x)
        y_min = min(y_min, y)
        y_max = max(y_max, y)
    
    box_face_rec = (y_min, x_max, y_max, x_min)
    bbox_mediapipe = [x_min, y_min, x_max, y_max]
    
    return bbox_mediapipe, box_face_rec

def get_3d_head_pose(face_landmarks, h, w):
    image_points_2d = np.array([
        (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h),
        (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h),
        (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h),
        (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h),
        (face_landmarks.landmark[287].x * w, face_landmarks.landmark[287].y * h),
        (face_landmarks.landmark[57].x * w, face_landmarks.landmark[57].y * h)
    ], dtype=np.float64)

    focal_length = w
    cam_center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, cam_center[0]],
        [0, focal_length, cam_center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    (success, rvec, tvec) = cv2.solvePnP(model_points_3d, image_points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    (axis_2d, _) = cv2.projectPoints(np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200]]).reshape(-1, 3), rvec, tvec, camera_matrix, dist_coeffs)
    R, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((R, tvec)))
    
    yaw = angles[1][0]
    pitch = angles[0][0]
    roll = angles[2][0]
    
    return pitch, yaw, roll, axis_2d.reshape(3, 2).astype(int), (int(image_points_2d[0][0]), int(image_points_2d[0][1]))


# --- HIGHLY OPTIMIZED: Live Stream Generator Function ---
def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}")
        return

    known_encodings = known_face_data["encodings"]
    known_names = known_face_data["names"]

    previous_objects = {}
    next_objectID = 0
    RECOGNITION_INTERVAL = 30 
    
    frame_counter = 0
    hand_bboxes = []
    
    with app.app_context():
        with mp_face_mesh.FaceMesh(
            max_num_faces=60,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, \
             mp_hands.Hands(
            max_num_hands=10,
            min_detection_confidence=0.7) as hands:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                    
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_counter += 1
                
                if frame_counter % HAND_DETECTION_SKIP_FRAMES == 0:
                    hand_bboxes = []
                    hand_results = hands.process(frame_rgb)
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            y_coords = [lm.y for lm in hand_landmarks.landmark]
                            hand_bboxes.append([int(min(x_coords) * w), int(min(y_coords) * h),
                                                int(max(x_coords) * w), int(max(y_coords) * h)])

                face_results = face_mesh.process(frame_rgb)
                
                current_frame_objects = []
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        bbox_mp, _ = get_bbox_and_centroid(face_landmarks.landmark, h, w)
                        centroid = ((bbox_mp[0] + bbox_mp[2]) // 2, (bbox_mp[1] + bbox_mp[3]) // 2)
                        current_frame_objects.append((centroid, face_landmarks, bbox_mp))
                
                current_objects = {} 

                if frame_counter % RECOGNITION_INTERVAL == 0:
                    for (centroid, face_landmarks, bbox_mp) in current_frame_objects:
                        box_face_rec = (bbox_mp[1], bbox_mp[2], bbox_mp[3], bbox_mp[0])
                        current_face_encoding = face_recognition.face_encodings(frame_rgb, [box_face_rec])
                        
                        name = "Unknown"
                        if current_face_encoding:
                            matches = face_recognition.compare_faces(known_encodings, current_face_encoding[0])
                            face_distances = face_recognition.face_distance(known_encodings, current_face_encoding[0])
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index]:
                                name = known_names[best_match_index]

                        best_match_id, min_dist = -1, 100 
                        
                        for objectID, (prev_centroid, prev_name, sleep_counter) in previous_objects.items():
                            distance = dist.euclidean(prev_centroid, centroid)
                            if distance < min_dist:
                                min_dist, best_match_id = distance, objectID
                        
                        if best_match_id != -1:
                            sleep_counter = previous_objects[best_match_id][2]
                            current_objects[best_match_id] = (centroid, name, sleep_counter)
                        else:
                            new_id = next_objectID
                            next_objectID += 1
                            current_objects[new_id] = (centroid, name, 0)
                else:
                    for (centroid, face_landmarks, bbox_mp) in current_frame_objects:
                        best_match_id, min_dist = -1, 100
                        
                        for objectID, (prev_centroid, prev_name, sleep_counter) in previous_objects.items():
                            distance = dist.euclidean(prev_centroid, centroid)
                            if distance < min_dist:
                                min_dist, best_match_id = distance, objectID
                        
                        if best_match_id != -1:
                            name = previous_objects[best_match_id][1]
                            sleep_counter = previous_objects[best_match_id][2]
                            current_objects[best_match_id] = (centroid, name, sleep_counter)
                        else:
                            new_id = next_objectID
                            next_objectID += 1
                            current_objects[new_id] = (centroid, "Unknown", 0)

                if not current_objects:
                    next_objectID = 0

                previous_objects = current_objects.copy()

                for objectID, (centroid, name, sleep_counter) in current_objects.items():
                    
                    found_landmarks = None
                    for (c, landmarks, _) in current_frame_objects:
                        if c == centroid:
                            found_landmarks = landmarks
                            break
                    
                    if not found_landmarks:
                        continue 
                    
                    bbox_mp, _ = get_bbox_and_centroid(found_landmarks.landmark, h, w)
                    is_covered = any(check_overlap(bbox_mp, hand_bbox) for hand_bbox in hand_bboxes)
                    
                    try:
                        pitch, yaw, roll, pose_axis, nose_tip = get_3d_head_pose(found_landmarks, h, w)
                    except Exception as e:
                        continue 

                    left_ear = calculate_ear(LEFT_EYE_INDICES, found_landmarks, (h, w))
                    right_ear = calculate_ear(RIGHT_EYE_INDICES, found_landmarks, (h, w))
                    ear = (left_ear + right_ear) / 2.0
                    
                    status = "Attentive"
                    if pitch > FOCUS_PITCH_THRESHOLD:
                        status = "Looking Down"
                    elif abs(yaw) > FOCUS_YAW_THRESHOLD:
                        status = "Looking Away"
                    
                    if ear < EAR_THRESHOLD:
                        sleep_counter += 1 
                        if sleep_counter >= EAR_CONSEC_FRAMES:
                            status = "Sleepy"
                    else:
                        sleep_counter = 0 
                    
                    previous_objects[objectID] = (centroid, name, sleep_counter)
                    if is_covered: status = "Face Covered"

                    attentiveness_score = SCORE_WEIGHTS.get(status, 0.0)
                    roll_no = name if name != "Unknown" else f"student_{objectID}"

                    log_data = {
                        "roll_no": roll_no,
                        "status": status,
                        "attentiveness_score": attentiveness_score
                    }
                    db_queue.put(log_data)
                    
                    color = (0, 255, 0) if status == "Attentive" else (0, 0, 255)
                    (x_min, y_min, x_max, y_max) = bbox_mp
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(frame, f"{roll_no}: {status}", (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.line(frame, nose_tip, tuple(pose_axis[0]), (0, 0, 255), 3)
                    cv2.line(frame, nose_tip, tuple(pose_axis[1]), (0, 255, 0), 3)
                    cv2.line(frame, nose_tip, tuple(pose_axis[2]), (255, 0, 0), 3)
                
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                if not flag:
                    continue
                
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encodedImage) + b'\r\n')

    cap.release()


# --- Flask Routes ---

# --- NEW: Welcome/Home Page Route ---
@app.route('/welcome')
def welcome():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    return render_template('welcome.html')

# --- UPDATED: / route now redirects to welcome or index ---
@app.route('/')
def root():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    else:
        return redirect(url_for('welcome'))

# --- NEW: /index route is now the main dashboard ---
@app.route('/index')
@login_required
def index():
    # --- 1. GET NEW DATA FOR DASHBOARD WIDGETS ---
    
    # Get Current Session Details
    current_session = None
    try:
        ist = pytz.timezone('Asia/Kolkata')
        # This will fail on 2025-11-14 11:41 AM (current time) -> 11
        # It's working
        current_hour_24 = datetime.now(ist).hour
        
        # Hour mapping (e.g., 9AM -> Hour 1)
        # This is based on your example timetable. Adjust as needed.
        hour_map = {
            8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9
            # Assuming 8:45-9:35 is "Hour 1" (which starts at 8)
        }
        # A simple 1-hour approximation
        class_hour = hour_map.get(current_hour_24) 
        
        if class_hour:
            entry = Timetable.query.filter_by(hour=class_hour).first()
            if entry and entry.subject != 'Not Set':
                current_session = {
                    "subject": entry.subject,
                    "teacher_name": entry.teacher_name,
                    # Example of formatting the hour nicely
                    "hour_pretty": f"{current_hour_24}:00 - {current_hour_24+1}:00" 
                }
    except Exception as e:
        print(f"Could not get current session: {e}")

    # --- 2. GET FILTERED DATA FOR MAIN PLOTS ---
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    query = AttentivenessLog.query
    
    if start_date_str:
        start_date = datetime.combine(date.fromisoformat(start_date_str), time.min)
        query = query.filter(AttentivenessLog.timestamp >= start_date)

    if end_date_str:
        end_date = datetime.combine(date.fromisoformat(end_date_str), time.max)
        query = query.filter(AttentivenessLog.timestamp <= end_date)

    try:
        db_queue.join() 
        data = pd.read_sql(query.statement, con=db.engine)
        
        if data.empty:
            return render_template('index.html', 
                                   totalStudents=0,
                                   current_session=current_session, # Still pass session info
                                   student_progress=None,
                                   graphJSON_line="{}", 
                                   graphJSON_box="{}",
                                   graphJSON_pie="{}")
            
    except Exception as e:
        flash(f'Error loading data: {e}', 'danger')
        return render_template('index.html', 
                               totalStudents=0,
                               current_session=current_session,
                               student_progress=None,
                               graphJSON_line="{}", 
                               graphJSON_box="{}",
                               graphJSON_pie="{}")

    # --- 3. CALCULATE ALL STATS AND GRAPHS ---

    # --- KPIs ---
    total_students = data['roll_no'].nunique()
    class_avg = f"{data['attentiveness_score'].mean() * 100:.1f}%"
    at_risk_df = data.groupby('roll_no')['attentiveness_score'].mean()
    at_risk_count = at_risk_df[at_risk_df < 0.6].count()

    # --- NEW: Student Progress List ---
    student_progress_df = data.groupby('roll_no')['attentiveness_score'].mean()
    student_progress_list = [
        {'name': index, 'score': value} 
        for index, value in student_progress_df.items()
    ]
    student_progress_list = sorted(student_progress_list, key=lambda x: x['score'], reverse=True)


    # --- Graph 1: Line Graph ---
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    line_df = data.set_index('timestamp').resample('20S')['attentiveness_score'].mean().reset_index()
    fig_line = px.line(line_df, x='timestamp', y='attentiveness_score',
                       title="Average Class Attentiveness Over Time")
    fig_line.update_layout(title=None, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphJSON_line = json.dumps(fig_line, cls=plotly.utils.PlotlyJSONEncoder)

    # --- Graph 2: Box Plot ---
    fig_box = px.box(data, x='roll_no', y='attentiveness_score', 
                     title="Student Attentiveness Distribution",
                     labels={'roll_no': 'Student', 'attentiveness_score': 'Score'})
    fig_box.update_layout(title=None, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    graphJSON_box = json.dumps(fig_box, cls=plotly.utils.PlotlyJSONEncoder)

    # --- NEW: Graph 3: Pie Chart ---
    status_counts = data['status'].value_counts()
    fig_pie = px.pie(status_counts, values=status_counts.values, names=status_counts.index,
                     title="Overall Status Distribution")
    fig_pie.update_layout(title=None, paper_bgcolor='rgba(0,0,0,0)')
    graphJSON_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)


    return render_template('index.html',
                           totalStudents=total_students,
                           classAvg=class_avg,
                           atRiskCount=at_risk_count,
                           graphJSON_line=graphJSON_line,
                           graphJSON_box=graphJSON_box,
                           graphJSON_pie=graphJSON_pie,           # <-- Pass new graph
                           current_session=current_session,       # <-- Pass new data
                           student_progress=student_progress_list # <-- Pass new data
                           )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and bcrypt.check_password_hash(user.password, request.form['password']):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        hashed_password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        new_user = User(username=request.form['username'], password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('welcome')) # <-- Redirect to welcome page

@app.route('/live_stream')
@login_required
def live_stream_page():
    return render_template('live_stream.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Timetable Route ---
@app.route('/timetable', methods=['GET', 'POST'])
@login_required
def timetable():
    if request.method == 'POST':
        for hour in range(1, 8): 
            subject = request.form.get(f'subject_{hour}')
            teacher_name = request.form.get(f'teacher_name_{hour}')
            teacher_email = request.form.get(f'teacher_email_{hour}')
            
            entry = Timetable.query.filter_by(hour=hour).first()
            if not entry:
                entry = Timetable(hour=hour)
                db.session.add(entry)
            
            entry.subject = subject if subject else 'Not Set'
            entry.teacher_name = teacher_name if teacher_name else 'Not Set'
            entry.teacher_email = teacher_email if teacher_email else 'Not Set'
            
        db.session.commit()
        flash('Timetable updated successfully!', 'success')
        return redirect(url_for('timetable'))

    schedule = Timetable.query.order_by(Timetable.hour).all()
    
    if len(schedule) < 7:
        for hour in range(1, 8):
            if not any(s.hour == hour for s in schedule):
                entry = Timetable(hour=hour)
                db.session.add(entry)
        db.session.commit()
        schedule = Timetable.query.order_by(Timetable.hour).all()

    return render_template('timetable.html', schedule=schedule)

# --- Send Report Route (Redirects to 'index') ---
@app.route('/send_report')
@login_required
def send_report():
    ist = pytz.timezone('Asia/Kolkata')
    current_hour_24 = datetime.now(ist).hour
    
    hour_map = {
        8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6, 14: 7, 15: 8, 16: 9
    }
    class_hour = hour_map.get(current_hour_24) 

    if not class_hour:
        flash(f'No class is scheduled for the current hour ({current_hour_24}:00). Cannot send report.', 'info')
        return redirect(url_for('index')) # <-- UPDATED

    timetable_entry = Timetable.query.filter_by(hour=class_hour).first()
    if not timetable_entry or timetable_entry.teacher_email == 'Not Set':
        flash(f'No teacher email found for Hour {class_hour}. Please update the timetable.', 'danger')
        return redirect(url_for('timetable'))
    
    teacher_email = timetable_entry.teacher_email
    teacher_name = timetable_entry.teacher_name
    subject = timetable_entry.subject

    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    query = AttentivenessLog.query
    report_date_str = date.today().isoformat()
    
    if start_date_str:
        start_date = datetime.combine(date.fromisoformat(start_date_str), time.min)
        query = query.filter(AttentivenessLog.timestamp >= start_date)
        report_date_str = start_date_str
    
    if end_date_str:
        end_date = datetime.combine(date.fromisoformat(end_date_str), time.max)
        query = query.filter(AttentivenessLog.timestamp <= end_date)
        if start_date_str:
            report_date_str = f"{start_date_str} to {end_date_str}"
        else:
            report_date_str = end_date_str

    try:
        db_queue.join()
        data = pd.read_sql(query.statement, con=db.engine)
        if data.empty:
            flash('No data to report for the selected filters.', 'info')
            return redirect(url_for('index', start_date=start_date_str, end_date=end_date_str)) # <-- UPDATED
    except Exception as e:
        flash(f'Error loading data for report: {e}', 'danger')
        return redirect(url_for('index', start_date=start_date_str, end_date=end_date_str)) # <-- UPDATED

    class_avg_str = f"{data['attentiveness_score'].mean() * 100:.1f}%"
    at_risk_df = data.groupby('roll_no')['attentiveness_score'].mean()
    at_risk_df = at_risk_df[at_risk_df < 0.6]
    at_risk_count_int = at_risk_df.count()

    at_risk_list = [
        {'name': index, 'score': value} 
        for index, value in at_risk_df.items()
    ]
    at_risk_list = sorted(at_risk_list, key=lambda x: x['score'])
    
    email_html = render_template('email_report.html',
                                 teacher_name=teacher_name,
                                 subject=subject,
                                 report_date=report_date_str,
                                 class_avg=class_avg_str,
                                 at_risk_count=at_risk_count_int,
                                 at_risk_list=at_risk_list
                                )
    
    msg = Message(
        f'Attentiveness Report for {subject} - {report_date_str}',
        sender=app.config['MAIL_USERNAME'],
        recipients=[teacher_email]
    )
    msg.html = email_html
    
    try:
        mail.send(msg)
        flash(f'Report successfully sent to {teacher_email}!', 'success')
    except Exception as e:
        flash(f'Error sending email: {e}', 'danger')

    return redirect(url_for('index', start_date=start_date_str, end_date=end_date_str)) # <-- UPDATED


# --- Clear Logs Route (Redirects to 'index') ---
@app.route('/clear_logs')
@login_required
def clear_logs():
    """
    Deletes all records from the AttentivenessLog table.
    """
    try:
        db_queue.join()
        num_rows_deleted = db.session.query(AttentivenessLog).delete()
        db.session.commit()
        flash(f'Successfully cleared {num_rows_deleted} log entries.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error clearing logs: {e}', 'danger')
        
    return redirect(url_for('index')) # <-- UPDATED


# --- Main Run ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
    app.run(debug=True, threaded=True, use_reloader=False)