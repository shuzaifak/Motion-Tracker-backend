import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from flask import Flask, request, jsonify, Response
import threading
import logging
import traceback
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
tracker = None
tracker_thread = None
tracker_lock = threading.Lock()
current_frame = None
running = False

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    response.headers.add('Cache-Control', 'no-store, no-cache, must-revalidate')
    return response

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RehabMotionTracker:
    def __init__(self):
        try:
            logger.info("Attempting to instantiate RehabMotionTracker")
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.debug("MediaPipe Pose initialized")

            # Storage
            self.data_dir = "rehab_data"
            os.makedirs(self.data_dir, exist_ok=True)

            # ROM history
            self.current_exercise = "Arm Raise"
            self.historical_max_rom = 0

            # Runtime
            self.cap = None  # Not used; camera managed by Flutter
            self.is_recording = False
            self.start_time = 0
            self.exercise_data = []
            self.recorded_landmarks = []
            self.rep_counter = 0
            self.last_position = "down"
            self.angle_threshold = 100
            self.pain_level = 0
            self.angle_history = []
            self.max_history_points = 100
            self.rep_timestamps = []
            self.last_rep_time = 0
            self.current_rep_speed = "Normal"
            self.session_max_rom = 0
            self.initialized = False

            # UI elements (mocked for compatibility)
            self.angle_label = MockLabel("0°")
            self.rom_label = MockLabel("0°")
            self.prev_rom_label = MockLabel("0°")
            self.rep_label = MockLabel("0")
            self.pain_label = MockLabel("0")
            self.time_label = MockLabel("00:00")
            self.speed_label = MockLabel("Normal")
            self.feedback_label = MockLabel("")
            self.status_label = MockLabel("Ready")
            self.record_btn = MockButton("Start Recording")
            self.rep_progress = MockProgress()
            self.exercise_var = MockVar(self.current_exercise)
            self.rep_goal_var = MockVar("10")
            self.graph_canvas = MockCanvas()
            self.ax = MockAxes()

            # Exercises
            self.exercises = {
                "Arm Raise": self.track_arm_raise,
                "Knee Bend": self.track_knee_bend,
                "Shoulder Rotation": self.track_shoulder_rotation,
                "Elbow Flexion": self.track_elbow_flexion,
                "Hip Abduction": self.track_hip_abduction,
                "Squat": self.track_squat,
                "Neck Rotation": self.track_neck_rotation,
                "Ankle Dorsiflexion": self.track_ankle_dorsiflexion,
                "Wrist Flexion": self.track_wrist_flexion,
                "Trunk Rotation": self.track_trunk_rotation,
                "Arm Raise Left": self.track_arm_raise_left,
                "Elbow Flexion Left": self.track_elbow_flexion_left,
                "Knee Bend Left": self.track_knee_bend_left,
                "Hip Abduction Left": self.track_hip_abduction_left,
                "Shoulder Rotation Left": self.track_shoulder_rotation_left,
                "Calf Raise": self.track_calf_raise,
                "Leg Extension": self.track_leg_extension,
                "Shoulder Abduction": self.track_shoulder_abduction,
                "Toe Raise": self.track_toe_raise,
                "Hip Extension": self.track_hip_extension,
            }
            logger.info("RehabMotionTracker instantiated successfully")
        except Exception as e:
            logger.error(f"Failed to instantiate RehabMotionTracker: {str(e)}\n{traceback.format_exc()}")
            raise

    def setup_ui(self):
        logger.debug("Skipping UI setup (headless mode)")
        self.load_historical_data()
        self.initialized = True
        logger.info("UI setup complete")

    def load_historical_data(self):
        try:
            self.historical_max_rom = 0
            files = [f for f in os.listdir(self.data_dir)
                     if f.startswith(self.current_exercise) and f.endswith('.csv')]
            if files:
                with open(os.path.join(self.data_dir, sorted(files)[-1]), 'r') as fh:
                    angles = [float(r['angle']) for r in csv.DictReader(fh)]
                    if angles:
                        self.historical_max_rom = max(angles)
            self.prev_rom_label.configure(text=f"{self.historical_max_rom:.1f}°")
            logger.debug(f"Loaded historical ROM: {self.historical_max_rom:.1f}°")
        except Exception as e:
            logger.error(f"History load error: {str(e)}\n{traceback.format_exc()}")
            self.prev_rom_label.configure(text="0°")

    def update_pain_level(self, value):
        try:
            self.pain_level = int(value)
            self.pain_label.configure(text=str(self.pain_level))
            if self.pain_level > 7:
                self.feedback_label.configure(text="WARNING: Pain level is high. Consider stopping exercise.", text_color="#ff0000")
            elif self.pain_level > 5:
                self.feedback_label.configure(text="Caution: Moderate pain detected. Consider reducing intensity.", text_color="#ff8800")
            elif self.pain_level > 3:
                self.feedback_label.configure(text="Note: Mild discomfort reported. Monitor closely.", text_color="#ffcc00")
            else:
                self.feedback_label.configure(text="", text_color="#ffd700")
            logger.debug(f"Pain level updated to: {self.pain_level}")
        except Exception as e:
            logger.error(f"Error updating pain level: {str(e)}\n{traceback.format_exc()}")

    def toggle_camera(self):
        self.status_label.configure(text="Camera state updated by client.")
        logger.debug("Camera toggle processed (Flutter-managed)")

    def toggle_recording(self):
        try:
            if not self.is_recording:
                self.is_recording = True
                self.start_time = time.time()
                self.exercise_data = []
                self.recorded_landmarks = []
                self.session_max_rom = 0
                self.record_btn.configure(text="Stop Recording")
                self.status_label.configure(text=f"Recording {self.current_exercise}...")
                self.prev_rom_label.configure(text=f"{self.historical_max_rom:.1f}°")
            else:
                self.is_recording = False
                self.record_btn.configure(text="Start Recording")
                self.status_label.configure(text="Recording stopped. Data saved.")
                self.save_exercise_data()
            logger.info(f"Recording {'started' if self.is_recording else 'stopped'}")
        except Exception as e:
            logger.error(f"Error toggling recording: {str(e)}\n{traceback.format_exc()}")

    def reset_counter(self):
        try:
            self.rep_counter = 0
            self.rep_label.configure(text=str(self.rep_counter))
            self.rep_progress.set(0)
            self.angle_history = []
            self.rep_timestamps = []
            self.session_max_rom = 0
            self.rom_label.configure(text="0°")
            self.ax.clear()
            self.ax.set_title("Joint Angle Over Time")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Angle (degrees)")
            self.ax.set_ylim(0, 180)
            self.ax.grid(True)
            self.graph_canvas.draw()
            self.feedback_label.configure(text="")
            self.status_label.configure(text=f"Counter reset. Ready for {self.current_exercise}.")
            logger.info("Counter reset")
        except Exception as e:
            logger.error(f"Error resetting counter: {str(e)}\n{traceback.format_exc()}")

    def change_exercise(self, exercise_name):
        try:
            self.current_exercise = exercise_name
            self.reset_counter()
            self.load_historical_data()
            self.exercise_var.set(exercise_name)
            self.status_label.configure(text=f"Exercise changed to {exercise_name}")
            if exercise_name in ["Arm Raise", "Elbow Flexion", "Arm Raise Left", "Elbow Flexion Left"]:
                self.last_position = "down"
            elif exercise_name in ["Knee Bend", "Squat", "Knee Bend Left"]:
                self.last_position = "straight"
            elif exercise_name in ["Hip Abduction", "Ankle Dorsiflexion", "Trunk Rotation", "Hip Abduction Left"]:
                self.last_position = "center"
            elif exercise_name in ["Wrist Flexion"]:
                self.last_position = "neutral"
            elif exercise_name in ["Neck Rotation"]:
                self.last_position = "center"
            elif exercise_name in ["Shoulder Rotation", "Shoulder Rotation Left"]:
                self.last_position = "down"
            elif exercise_name in ["Calf Raise", "Toe Raise"]:
                self.last_position = "center"
            elif exercise_name in ["Leg Extension"]:
                self.last_position = "bent"
            elif exercise_name in ["Shoulder Abduction"]:
                self.last_position = "down"
            elif exercise_name in ["Hip Extension"]:
                self.last_position = "center"
            if exercise_name in ["Arm Raise", "Arm Raise Left"]:
                self.angle_threshold = 100
            elif exercise_name in ["Knee Bend", "Squat", "Knee Bend Left"]:
                self.angle_threshold = 120
            elif exercise_name in ["Hip Abduction", "Hip Abduction Left"]:
                self.angle_threshold = 20
            elif exercise_name in ["Ankle Dorsiflexion", "Toe Raise"]:
                self.angle_threshold = 15
            elif exercise_name in ["Wrist Flexion"]:
                self.angle_threshold = 30
            elif exercise_name in ["Trunk Rotation"]:
                self.angle_threshold = 45
            elif exercise_name in ["Neck Rotation"]:
                self.angle_threshold = 40
            elif exercise_name in ["Shoulder Rotation", "Shoulder Rotation Left"]:
                self.angle_threshold = 90
            elif exercise_name in ["Elbow Flexion", "Elbow Flexion Left"]:
                self.angle_threshold = 70
            elif exercise_name in ["Calf Raise"]:
                self.angle_threshold = 15
            elif exercise_name in ["Leg Extension"]:
                self.angle_threshold = 160
            elif exercise_name in ["Shoulder Abduction"]:
                self.angle_threshold = 100
            elif exercise_name in ["Hip Extension"]:
                self.angle_threshold = 20
            logger.info(f"Exercise changed to: {exercise_name}, threshold: {self.angle_threshold}")
        except Exception as e:
            logger.error(f"Error changing exercise: {str(e)}\n{traceback.format_exc()}")

    def update_graph(self):
        try:
            self.ax.clear()
            self.ax.plot(range(len(self.angle_history)), self.angle_history, 'b-')
            self.ax.set_title("Joint Angle Over Time")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Angle (°)")
            self.ax.set_ylim(0, 180)
            self.ax.grid(True)
            self.graph_canvas.draw()
            logger.debug("Graph updated")
        except Exception as e:
            logger.error(f"Error updating graph: {str(e)}\n{traceback.format_exc()}")

    def save_exercise_data(self):
        try:
            if self.exercise_data:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(self.data_dir, f"{self.current_exercise}_{timestamp}.csv")
                with open(filename, 'w', newline='') as csvfile:
                    fieldnames = ['time', 'angle', 'rep_count', 'pain_level']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for data in self.exercise_data:
                        writer.writerow(data)
                self.load_historical_data()
                logger.info(f"Data saved to: {filename}")
        except Exception as e:
            logger.error(f"Error saving exercise data: {str(e)}\n{traceback.format_exc()}")

    def calculate_angle(self, a, b, c):
        try:
            a, b, c = np.array([a.x, a.y]), np.array([b.x, b.y]), np.array([c.x, c.y])
            ang = np.degrees(abs(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])))
            angle = 360 - ang if ang > 180 else ang
            return angle
        except Exception as e:
            logger.error(f"Error calculating angle: {str(e)}\n{traceback.format_exc()}")
            return 0.0

    def compute_joint_ranges(self, landmarks_list):
        try:
            joint_ranges = {}
            tracked_joints = [
                'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW',
                'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP',
                'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE'
            ]
            for joint in tracked_joints:
                joint_ranges[joint] = {'x_min': float('inf'), 'x_max': float('-inf'), 'y_min': float('inf'), 'y_max': float('-inf')}
            for landmarks in landmarks_list:
                for joint_name in tracked_joints:
                    idx = getattr(self.mp_pose.PoseLandmark, joint_name)
                    lm = landmarks.landmark[idx]
                    joint_ranges[joint_name]['x_min'] = min(joint_ranges[joint_name]['x_min'], lm.x)
                    joint_ranges[joint_name]['x_max'] = max(joint_ranges[joint_name]['x_max'], lm.x)
                    joint_ranges[joint_name]['y_min'] = min(joint_ranges[joint_name]['y_min'], lm.y)
                    joint_ranges[joint_name]['y_max'] = max(joint_ranges[joint_name]['y_max'], lm.y)
            for joint in joint_ranges:
                joint_ranges[joint]['x_range'] = joint_ranges[joint]['x_max'] - joint_ranges[joint]['x_min']
                joint_ranges[joint]['y_range'] = joint_ranges[joint]['y_max'] - joint_ranges[joint]['y_min']
            return joint_ranges
        except Exception as e:
            logger.error(f"Error computing joint ranges: {str(e)}\n{traceback.format_exc()}")
            return {}

    def track_rep_speed(self):
        try:
            current_time = time.time()
            self.rep_timestamps.append(current_time)
            if len(self.rep_timestamps) > 1:
                interval = self.rep_timestamps[-1] - self.rep_timestamps[-2]
                if interval < 0.5:
                    self.current_rep_speed = "Fast"
                elif interval > 1.5:
                    self.current_rep_speed = "Slow"
                else:
                    self.current_rep_speed = "Normal"
                self.speed_label.configure(text=self.current_rep_speed)
                if self.current_rep_speed == "Fast":
                    self.feedback_label.configure(text="Slow down for better control.", text_color="#ff8800")
                elif self.current_rep_speed == "Slow":
                    self.feedback_label.configure(text="Try to increase speed slightly.", text_color="#ff8800")
            logger.debug(f"Rep speed updated: {self.current_rep_speed}")
        except Exception as e:
            logger.error(f"Error tracking rep speed: {str(e)}\n{traceback.format_exc()}")

    def check_exercise_form(self, landmarks, exercise):
        try:
            feedback = ""
            if exercise == "arm_raise":
                hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                if abs(hip.y - shoulder.y) > 0.1:
                    feedback = "Keep shoulders level with hips."
            elif exercise == "knee_bend":
                knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
                if knee.x < ankle.x:
                    feedback = "Keep knees over ankles."
            elif exercise == "squat":
                hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
                knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
                if hip.y < knee.y:
                    feedback = "Lower hips further for deeper squat."
            if feedback:
                self.feedback_label.configure(text=feedback, text_color="#ff8800")
                logger.debug(f"Form feedback: {feedback}")
        except Exception as e:
            logger.error(f"Error checking exercise form: {str(e)}\n{traceback.format_exc()}")

    def track_arm_raise(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            angle = self.calculate_angle(hip, shoulder, elbow)
            if angle > self.angle_threshold and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
                self.check_exercise_form(landmarks, "arm_raise")
            elif angle < 70 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking arm raise: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_arm_raise_left(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            angle = self.calculate_angle(hip, shoulder, elbow)
            if angle > self.angle_threshold and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle < 70 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking arm raise left: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_knee_bend(self, landmarks, frame):
        try:
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            angle = self.calculate_angle(hip, knee, ankle)
            if angle < 120 and self.last_position == "straight":
                self.last_position = "bent"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
                self.check_exercise_form(landmarks, "knee_bend")
            elif angle > 160 and self.last_position == "bent":
                self.last_position = "straight"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking knee bend: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_knee_bend_left(self, landmarks, frame):
        try:
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            angle = self.calculate_angle(hip, knee, ankle)
            if angle < 120 and self.last_position == "straight":
                self.last_position = "bent"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle > 160 and self.last_position == "bent":
                self.last_position = "straight"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking knee bend left: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_shoulder_rotation(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            angle = self.calculate_angle(shoulder, elbow, wrist)
            wrist_y_normalized = (wrist.y - elbow.y) * frame.shape[0]
            if wrist_y_normalized < -30 and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif wrist_y_normalized > 30 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking shoulder rotation: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_shoulder_rotation_left(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            angle = self.calculate_angle(shoulder, elbow, wrist)
            wrist_y_normalized = (wrist.y - elbow.y) * frame.shape[0]
            if wrist_y_normalized < -30 and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif wrist_y_normalized > 30 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking shoulder rotation left: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_elbow_flexion(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            angle = self.calculate_angle(shoulder, elbow, wrist)
            if angle < 70 and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle > 150 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking elbow flexion: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_elbow_flexion_left(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            angle = self.calculate_angle(shoulder, elbow, wrist)
            if angle < 70 and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle > 150 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking elbow flexion left: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_hip_abduction(self, landmarks, frame):
        try:
            hip_left = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            hip_right = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            hip_mid_x = (hip_left.x + hip_right.x) / 2
            hip_mid_y = (hip_left.y + hip_right.y) / 2
            v1 = [hip_right.x - hip_mid_x, hip_right.y - hip_mid_y]
            v2 = [knee.x - hip_right.x, knee.y - hip_right.y]
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2) + 1e-10
            magnitude_v2 = np.sqrt(v2[0]**2 + v2[1]**2) + 1e-10
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = np.degrees(np.arccos(cos_angle))
            display_angle = 90 - angle if angle < 90 else angle - 90
            if display_angle > self.angle_threshold and self.last_position == "center":
                self.last_position = "side"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif display_angle < 10 and self.last_position == "side":
                self.last_position = "center"
            cv2.putText(frame, f"{display_angle:.1f}°",
                        (int(hip_right.x * frame.shape[1]), int(hip_right.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return display_angle
        except Exception as e:
            logger.error(f"Error tracking hip abduction: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_hip_abduction_left(self, landmarks, frame):
        try:
            hip_right = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            hip_left = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            hip_mid_x = (hip_left.x + hip_right.x) / 2
            hip_mid_y = (hip_left.y + hip_right.y) / 2
            v1 = [hip_left.x - hip_mid_x, hip_left.y - hip_mid_y]
            v2 = [knee.x - hip_left.x, knee.y - hip_left.y]
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude_v1 = np.sqrt(v1[0]**2 + v1[1]**2) + 1e-10
            magnitude_v2 = np.sqrt(v2[0]**2 + v2[1]**2) + 1e-10
            cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
            cos_angle = max(-1, min(1, cos_angle))
            angle = np.degrees(np.arccos(cos_angle))
            display_angle = 90 - angle if angle < 90 else angle - 90
            if display_angle > self.angle_threshold and self.last_position == "center":
                self.last_position = "side"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif display_angle < 10 and self.last_position == "side":
                self.last_position = "center"
            cv2.putText(frame, f"{display_angle:.1f}°",
                        (int(hip_left.x * frame.shape[1]), int(hip_left.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return display_angle
        except Exception as e:
            logger.error(f"Error tracking hip abduction left: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_squat(self, landmarks, frame):
        try:
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            angle = self.calculate_angle(hip, knee, ankle)
            if angle < 120 and self.last_position == "straight":
                self.last_position = "squat"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
                self.check_exercise_form(landmarks, "squat")
            elif angle > 160 and self.last_position == "squat":
                self.last_position = "straight"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking squat: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_neck_rotation(self, landmarks, frame):
        try:
            nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            shoulder_left = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
            shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
            angle = np.degrees(np.arctan2(nose.x - shoulder_mid_x, nose.y - shoulder_mid_y))
            abs_angle = abs(angle)
            if abs_angle > self.angle_threshold and self.last_position == "center":
                self.last_position = "turned"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif abs_angle < 10 and self.last_position == "turned":
                self.last_position = "center"
            cv2.putText(frame, f"{abs_angle:.1f}°",
                        (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return abs_angle
        except Exception as e:
            logger.error(f"Error tracking neck rotation: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_ankle_dorsiflexion(self, landmarks, frame):
        try:
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            foot = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            angle = self.calculate_angle(knee, ankle, foot)
            deviation = abs(angle - 90)
            if deviation > self.angle_threshold and self.last_position == "center":
                self.last_position = "flexed"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif deviation < 5 and self.last_position == "flexed":
                self.last_position = "center"
            cv2.putText(frame, f"{deviation:.1f}°",
                        (int(ankle.x * frame.shape[1]), int(ankle.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return deviation
        except Exception as e:
            logger.error(f"Error tracking ankle dorsiflexion: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_wrist_flexion(self, landmarks, frame):
        try:
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            hand = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_INDEX]
            angle = self.calculate_angle(elbow, wrist, hand)
            if angle > self.angle_threshold and self.last_position == "neutral":
                self.last_position = "flexed"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle < 10 and self.last_position == "flexed":
                self.last_position = "neutral"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking wrist flexion: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_trunk_rotation(self, landmarks, frame):
        try:
            shoulder_left = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            shoulder_right = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip_left = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            shoulder_mid_x = (shoulder_left.x + shoulder_right.x) / 2
            shoulder_mid_y = (shoulder_left.y + shoulder_right.y) / 2
            angle = np.degrees(np.arctan2(shoulder_right.x - shoulder_mid_x, hip_left.y - shoulder_mid_y))
            abs_angle = abs(angle)
            if abs_angle > self.angle_threshold and self.last_position == "center":
                self.last_position = "rotated"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif abs_angle < 10 and self.last_position == "rotated":
                self.last_position = "center"
            cv2.putText(frame, f"{abs_angle:.1f}°",
                        (int(shoulder_mid_x * frame.shape[1]), int(shoulder_mid_y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return abs_angle
        except Exception as e:
            logger.error(f"Error tracking trunk rotation: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_calf_raise(self, landmarks, frame):
        try:
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            foot = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
            angle = self.calculate_angle(knee, ankle, foot)
            deviation = abs(angle - 90)
            if deviation > self.angle_threshold and self.last_position == "center":
                self.last_position = "raised"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif deviation < 5 and self.last_position == "raised":
                self.last_position = "center"
            cv2.putText(frame, f"{deviation:.1f}°",
                        (int(ankle.x * frame.shape[1]), int(ankle.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return deviation
        except Exception as e:
            logger.error(f"Error tracking calf raise: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_leg_extension(self, landmarks, frame):
        try:
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            angle = self.calculate_angle(hip, knee, ankle)
            if angle > 160 and self.last_position == "bent":
                self.last_position = "straight"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle < 120 and self.last_position == "straight":
                self.last_position = "bent"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(knee.x * frame.shape[1]), int(knee.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking leg extension: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_shoulder_abduction(self, landmarks, frame):
        try:
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            angle = self.calculate_angle(hip, shoulder, elbow)
            if angle > self.angle_threshold and self.last_position == "down":
                self.last_position = "up"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle < 50 and self.last_position == "up":
                self.last_position = "down"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking shoulder abduction: {str(e)}\n{traceback.format_exc()}")
            return None

    def track_toe_raise(self, landmarks, frame):
        return self.track_ankle_dorsiflexion(landmarks, frame)

    def track_hip_extension(self, landmarks, frame):
        try:
            shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            knee = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            angle = self.calculate_angle(shoulder, hip, knee)
            if angle > self.angle_threshold and self.last_position == "center":
                self.last_position = "extended"
                self.rep_counter += 1
                self.rep_label.configure(text=str(self.rep_counter))
                self.track_rep_speed()
            elif angle < 10 and self.last_position == "extended":
                self.last_position = "center"
            cv2.putText(frame, f"{angle:.1f}°",
                        (int(hip.x * frame.shape[1]), int(hip.y * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            return angle
        except Exception as e:
            logger.error(f"Error tracking hip extension: {str(e)}\n{traceback.format_exc()}")
            return None

# Mock classes for UI compatibility
class MockLabel:
    def __init__(self, text):
        self._text = text
    def configure(self, **kwargs):
        if 'text' in kwargs:
            self._text = kwargs['text']
    def cget(self, key):
        if key == "text":
            return self._text
        return ""

class MockButton:
    def __init__(self, text):
        self._text = text
    def configure(self, **kwargs):
        if 'text' in kwargs:
            self._text = kwargs['text']

class MockProgress:
    def set(self, value):
        pass

class MockVar:
    def __init__(self, value):
        self._value = value
    def set(self, value):
        self._value = value
    def get(self):
        return self._value

class MockCanvas:
    def draw(self):
        pass

class MockAxes:
    def clear(self):
        pass
    def plot(self, x, y, fmt):
        pass
    def set_title(self, title):
        pass
    def set_xlabel(self, label):
        pass
    def set_ylabel(self, label):
        pass
    def set_ylim(self, ymin, ymax):
        pass
    def grid(self, b):
        pass

def run_tracker():
    global tracker, current_frame, running
    logger.info("Starting tracker thread")
    
    try:
        with tracker_lock:
            tracker = RehabMotionTracker()
            tracker.setup_ui()
            tracker.initialized = True
            running = True
            logger.info("Tracker successfully initialized")
            
        while running:
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Tracker error: {str(e)}\n{traceback.format_exc()}")
        with tracker_lock:
            tracker = None
            running = False

@app.route('/video_feed')
def video_feed():
    response = Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'Pragma': 'no-cache',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
    )
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

def generate_frames():
    test_pattern = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_pattern, "TEST PATTERN", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    while True:
        try:
            # Use test pattern if no camera frame available
            frame = current_frame if current_frame is not None else test_pattern
            
            # Add timestamp for debugging
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to JPEG with quality settings
            ret, buffer = cv2.imencode('.jpg', frame, [
                int(cv2.IMWRITE_JPEG_QUALITY), 80,
                int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
            ])
            
            if not ret:
                print("Frame encoding failed")
                continue
                
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + 
                  buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)  # ~30fps
            
        except Exception as e:
            print(f"Error in frame generation: {str(e)}")
            time.sleep(1)
            
@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global tracker, current_frame
    logger.info(f"Received upload_frame request from {request.remote_addr}, headers: {request.headers}")
    try:
        with tracker_lock:
            if not tracker or not tracker.initialized:
                logger.error("Tracker not initialized")
                return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
        if 'frame' not in request.files:
            logger.error("No frame in request")
            return jsonify({"status": "error", "message": "No frame provided"}), 400
        file = request.files['frame']
        logger.debug(f"Frame file received, size: {file.content_length} bytes")
        nparr = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logger.error("Invalid frame received")
            return jsonify({"status": "error", "message": "Invalid frame"}), 400
        logger.info(f"Frame decoded, shape: {frame.shape}")
        
        # Process frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.pose.process(rgb_frame)
        logger.debug(f"Pose processed, landmarks detected: {results.pose_landmarks is not None}")
        
        if results.pose_landmarks:
            logger.debug("Drawing landmarks on frame")
            tracker.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                tracker.mp_pose.POSE_CONNECTIONS,
                tracker.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                tracker.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            tracking_function = tracker.exercises.get(tracker.current_exercise)
            if tracking_function:
                logger.debug(f"Tracking exercise: {tracker.current_exercise}")
                angle = tracking_function(results.pose_landmarks, frame)
                if angle is not None:
                    logger.debug(f"Angle calculated: {angle:.1f}°")
                    tracker.angle_label.configure(text=f"{angle:.1f}°")
                    if angle > tracker.session_max_rom:
                        tracker.session_max_rom = angle
                        tracker.rom_label.configure(text=f"{tracker.session_max_rom:.1f}°")
                    if tracker.is_recording:
                        elapsed_time = time.time() - tracker.start_time
                        tracker.exercise_data.append({
                            'time': elapsed_time,
                            'angle': angle,
                            'rep_count': tracker.rep_counter,
                            'pain_level': tracker.pain_level
                        })
                        mins, secs = divmod(int(elapsed_time), 60)
                        tracker.time_label.configure(text=f"{mins:02d}:{secs:02d}")
                        tracker.angle_history.append(angle)
                        if len(tracker.angle_history) > tracker.max_history_points:
                            tracker.angle_history.pop(0)
                        tracker.update_graph()
                        try:
                            goal = int(tracker.rep_goal_var.get())
                            if goal > 0:
                                progress = min(1.0, tracker.rep_counter / goal)
                                tracker.rep_progress.set(progress)
                                logger.debug(f"Progress updated: {progress:.2f}")
                        except ValueError as ve:
                            logger.warning(f"Invalid rep goal: {ve}")
            if tracker.is_recording:
                tracker.recorded_landmarks.append(results.pose_landmarks)
        
        with tracker_lock:
            current_frame = frame
            logger.debug("Frame stored in current_frame")
        
        if results.pose_landmarks:
            landmarks = [
                {"x": lm.x, "y": lm.y, "visibility": lm.visibility}
                for lm in results.pose_landmarks.landmark
            ]
            logger.info("Sending response with landmarks")
            return jsonify({"status": "success", "landmarks": landmarks})
        logger.info("Sending response without landmarks")
        return jsonify({"status": "success"})
    except Exception as e:
        logger.error(f"Frame processing error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Frame processing error: {str(e)}"}), 500

@app.route('/start_tracker', methods=['POST'])
def start_tracker():
    global tracker_thread
    logger.info(f"Received start_tracker request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker_thread is None or not tracker_thread.is_alive():
                tracker_thread = threading.Thread(target=run_tracker)
                tracker_thread.daemon = True
                tracker_thread.start()
                time.sleep(1)
                logger.info("Tracker thread started")
                return jsonify({"status": "success", "message": "Tracker started"})
            logger.warning("Tracker already running")
            return jsonify({"status": "success", "message": "Tracker already running"})
    except Exception as e:
        logger.error(f"Start tracker error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Start tracker error: {str(e)}"}), 500

@app.route('/stop_tracker', methods=['POST'])
def stop_tracker():
    global tracker, running
    logger.info(f"Received stop_tracker request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker:
                running = False
                tracker = None
                if tracker_thread is not None:
                    tracker_thread.join(timeout=2)
                    tracker_thread = None
                logger.info("Tracker stopped")
                return jsonify({"status": "success", "message": "Tracker stopped"})
            logger.warning("No tracker running")
            return jsonify({"status": "error", "message": "No tracker running"}), 400
    except Exception as e:
        logger.error(f"Stop tracker error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Stop tracker error: {str(e)}"}), 500

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    logger.info(f"Received toggle_camera request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                tracker.toggle_camera()
                logger.debug("Camera toggle processed (Flutter-managed)")
                return jsonify({
                    "status": "success",
                    "is_camera_on": True
                })
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Camera toggle error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Camera error: {str(e)}"}), 500

@app.route('/toggle_recording', methods=['POST'])
def toggle_recording():
    logger.info(f"Received toggle_recording request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                tracker.toggle_recording()
                return jsonify({
                    "status": "success",
                    "is_recording": tracker.is_recording
                })
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Recording error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Recording error: {str(e)}"}), 500

@app.route('/reset_counter', methods=['POST'])
def reset_counter():
    logger.info(f"Received reset_counter request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                tracker.reset_counter()
                return jsonify({"status": "success"})
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Reset counter error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Reset error: {str(e)}"}), 500

@app.route('/change_exercise', methods=['POST'])
def change_exercise():
    logger.info(f"Received change_exercise request from {request.remote_addr}, body: {request.get_json()}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                exercise = request.json.get('exercise')
                if exercise in tracker.exercises:
                    tracker.change_exercise(exercise)
                    return jsonify({"status": "success"})
                logger.error(f"Invalid exercise: {exercise}")
                return jsonify({"status": "error", "message": "Invalid exercise"}), 400
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Exercise change error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Exercise change error: {str(e)}"}), 500

@app.route('/set_pain_level', methods=['POST'])
def set_pain_level():
    logger.info(f"Received set_pain_level request from {request.remote_addr}, body: {request.get_json()}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                pain_level = request.json.get('pain_level')
                if pain_level is not None and 0 <= pain_level <= 10:
                    tracker.update_pain_level(pain_level)
                    return jsonify({"status": "success"})
                logger.error(f"Invalid pain level: {pain_level}")
                return jsonify({"status": "error", "message": "Invalid pain level"}), 400
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Pain level error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Pain level error: {str(e)}"}), 500

@app.route('/set_rep_goal', methods=['POST'])
def set_rep_goal():
    logger.info(f"Received set_rep_goal request from {request.remote_addr}, body: {request.get_json()}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                rep_goal = request.json.get('rep_goal')
                if rep_goal is not None and rep_goal > 0:
                    tracker.rep_goal_var.set(str(rep_goal))
                    return jsonify({"status": "success"})
                logger.error(f"Invalid rep goal: {rep_goal}")
                return jsonify({"status": "error", "message": "Invalid rep goal"}), 400
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Rep goal error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Rep goal error: {str(e)}"}), 500

@app.route('/get_status', methods=['GET', 'OPTIONS'])
def get_status():
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                return jsonify({
                    "status": "success",
                    "data": {
                        "exercise": tracker.current_exercise,
                        "rep_count": tracker.rep_counter,
                        "current_angle": 0.0, 
                        "max_rom": 0.0,      
                        "historical_max_rom": tracker.historical_max_rom,
                        "rep_goal": int(tracker.rep_goal_var.get()),
                        "pain_level": tracker.pain_level,
                        "is_recording": tracker.is_recording,
                        "is_camera_on": True,
                        "rep_speed": tracker.current_rep_speed,
                        "feedback": tracker.feedback_label.cget("text")
                    }
                }), 200
            return jsonify({"status": "error", "message": "Tracker not ready"}), 503
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route('/get_exercises', methods=['GET'])
def get_exercises():
    logger.info(f"Received get_exercises request from {request.remote_addr}")
    max_wait = 10  # seconds
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        with tracker_lock:
            if tracker is not None and tracker.initialized:
                exercises = list(tracker.exercises.keys())
                return jsonify({
                    "status": "success", 
                    "exercises": exercises,
                    "tracker_status": "initialized"
                })
            elif tracker is None:
                # Start tracker if not running
                start_tracker()
        
        time.sleep(0.5)
    
    return jsonify({
        "status": "error",
        "message": "Tracker initialization timed out",
        "tracker_status": "uninitialized"
    }), 500
    
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "server": "Motion Tracker",
        "time": datetime.now().isoformat()
    })
    
@app.route('/get_exercise_data', methods=['GET'])
def get_exercise_data():
    logger.info(f"Received get_exercise_data request from {request.remote_addr}")
    try:
        with tracker_lock:
            if tracker and tracker.initialized:
                exercise_data = tracker.exercise_data
                joint_ranges = tracker.compute_joint_ranges(tracker.recorded_landmarks)
                response_data = {
                    "status": "success",
                    "data": exercise_data,
                    "jointRanges": joint_ranges
                }
                logger.debug(f"Exercise data response: {response_data}")
                tracker.exercise_data = []
                tracker.recorded_landmarks = []
                return jsonify(response_data)
            logger.error("Tracker not initialized")
            return jsonify({"status": "error", "message": "Tracker not initialized"}), 500
    except Exception as e:
        logger.error(f"Exercise data error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"status": "error", "message": f"Exercise data error: {str(e)}"}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)