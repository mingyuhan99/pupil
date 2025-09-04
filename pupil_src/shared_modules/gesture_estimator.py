"""
Pupil Capture World Process Plugin for Eye Gesture Estimation
with Blink Calibration based on eyelid landmarks.
"""
import os
import time
from collections import deque
import numpy as np
import zmq
import msgpack
from plugin import Plugin
from pyglui import ui
import pickle
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Helper function for deserializing msgpack messages with unicode keys
def msgpack_unpackb_unicode(x):
    return msgpack.unpackb(x, raw=False)

# The gesture prediction model class (remains the same)
class M2EyeGesturePredictor:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.package = None
        self.load_model()

    def load_model(self):
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"Gesture model path is invalid: {self.model_path}")
            self.package = None
            return
        try:
            logger.info(f"Loading gesture model from: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self.package = pickle.load(f)
            self.classifier = self.package['components']['classifier']
            self.scaler = self.package['components']['scaler']
            self.landmark_weights = self.package['constants']['LANDMARK_WEIGHTS']
            logger.info("Gesture model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load gesture model: {e}")
            self.package = None

    def extract_delta_features(self, landmarks_sequence):
        if len(landmarks_sequence) != 5: return None
        landmarks_array = np.array(landmarks_sequence)
        deltas = np.diff(landmarks_array, axis=0)
        cumulative_delta = np.cumsum(deltas, axis=0)
        norm_delta = np.linalg.norm(deltas) + 1e-6
        norm_cumulative = np.linalg.norm(cumulative_delta) + 1e-6
        normalized_deltas = deltas / norm_delta
        normalized_cumulative = cumulative_delta / norm_cumulative
        weighted_deltas = normalized_deltas * self.landmark_weights
        weighted_cumulative = cumulative_delta * self.landmark_weights
        return np.concatenate([weighted_deltas.flatten(), weighted_cumulative.flatten()])

    def predict(self, eye0_landmarks, eye1_landmarks):
        if not self.package: return "Model not loaded"
        try:
            features0 = self.extract_delta_features(eye0_landmarks)
            features1 = self.extract_delta_features(eye1_landmarks)
            if features0 is None or features1 is None: return "Invalid data"

            fusion_features = np.concatenate([features0, features1])
            feature_magnitudes = [np.linalg.norm(features0), np.linalg.norm(features1)]
            max_mag, min_mag = max(feature_magnitudes), min(feature_magnitudes)
            magnitude_ratio = max_mag / (min_mag + 1e-6)
            direction_consistency = 1.0
            conflict_score = np.std(feature_magnitudes) / (np.mean(feature_magnitudes) + 1e-6)

            enhanced_features = np.concatenate([
                fusion_features,
                [direction_consistency, magnitude_ratio, conflict_score]
            ]).reshape(1, -1)

            features_scaled = self.scaler.transform(enhanced_features)
            prediction = self.classifier.predict(features_scaled)[0]

            if prediction == 'left': return 'right'
            if prediction == 'right': return 'left'
            return prediction
        except Exception as e:
            logger.error(f"Gesture prediction failed: {e}")
            if 'enhanced_features' in locals():
                logger.error(f"Feature dimension mismatch. Generated: {enhanced_features.shape[1]}, Expected: {self.scaler.n_features_in_}")
            return "Prediction Error"

class Gesture_Estimator(Plugin):
    uniqueness = "by_class"
    order = 0.6

    def __init__(self, g_pool):
        super().__init__(g_pool)
        # --- Settings ---
        self.enabled = False
        self.model_path = ""
        self.min_duration_ms = 400
        self.trim_ms = 100
        self.num_samples = 5
        self.cooldown_ms = 500
        # --- Calibration ---
        self.calib_duration_sec = 2.0
        self.calibrated_threshold = 55.0
        self.avg_open_score = 0.0
        self.avg_closed_score = 0.0
        self.calib_samples = []
        self.calib_start_time = 0.0
        # --- State ---
        self.state = 'IDLE' # IDLE, COLLECTING, COOLDOWN, CALIBRATING_OPEN, CALIBRATING_CLOSED
        self.buffer = {'eye0': deque(), 'eye1': deque()}
        self.eye_landmarks = {'eye0': None, 'eye1': None}
        self.start_time = 0.0
        self.cooldown_end_time = 0.0
        self.last_gesture = "None"
        self.predictor = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # --- UI ---
        self.gesture_info_text = None
        self.calib_status_text = None

        self.context = zmq.Context()
        self.pupil_sub = self.context.socket(zmq.SUB)
        self.pupil_sub.connect(self.g_pool.ipc_sub_url)
        self.pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.")

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Gesture Estimator"
        self.menu.append(ui.Switch('enabled', self, label="Enable Gesture Estimation"))
        self.menu.append(ui.Text_Input('model_path', self, label="Gesture Model Path (.pkl)"))
        self.menu.append(ui.Button("Load Model", self.load_model))
        
        # Calibration UI
        calib_menu = ui.Growing_Menu("Calibration")
        self.calib_status_text = ui.Info_Text("Status: Not Calibrated")
        calib_menu.append(self.calib_status_text)
        calib_menu.append(ui.Button("Start Blink Calibration", self.start_calibration))
        self.menu.append(calib_menu)

        # Settings UI
        settings_menu = ui.Growing_Menu("Advanced Settings")
        settings_menu.append(ui.Slider('min_duration_ms', self, min=200, max=1000, step=10, label="Min Duration (ms)"))
        settings_menu.append(ui.Slider('trim_ms', self, min=0, max=200, step=10, label="Trim Duration (ms)"))
        self.menu.append(settings_menu)
        
        self.gesture_info_text = ui.Info_Text(f"Last Gesture: {self.last_gesture}")
        self.menu.append(self.gesture_info_text)

    def _calculate_eye_aperture(self, landmarks):
        if landmarks is None or len(landmarks) != 8:
            return 0.0
        
        # Calculate vertical distances between opposing eyelid landmarks
        dist1 = abs(landmarks[0][1] - landmarks[7][1]) # Point 1 and 8
        dist2 = abs(landmarks[1][1] - landmarks[6][1]) # Point 2 and 7
        dist3 = abs(landmarks[2][1] - landmarks[5][1]) # Point 3 and 6
        
        return dist1 + dist2 + dist3

    def start_calibration(self):
        if self.state != 'IDLE':
            logger.warning("Cannot start calibration while in another state.")
            return
        self.state = 'CALIBRATING_OPEN'
        self.calib_start_time = time.time()
        self.calib_samples = []
        self.calib_status_text.text = "CALIBRATING: Please keep eyes OPEN and look forward."
        logger.info("Starting calibration: Eyes open.")

    def load_model(self):
        self.predictor = M2EyeGesturePredictor(self.model_path, self.device)

    def recent_events(self, events):
        try:
            while True:
                topic, payload = self.pupil_sub.recv_multipart(flags=zmq.NOBLOCK)
                pupil_data = msgpack_unpackb_unicode(payload)
                topic_str = topic.decode()

                if topic_str.startswith('pupil.'):
                    try:
                        parts = topic_str.split('.')
                        if len(parts) > 1 and parts[1].isdigit():
                            eye_id = int(parts[1])
                            eye_id_str = f"eye{eye_id}"
                            if eye_id_str in self.eye_landmarks:
                                self.eye_landmarks[eye_id_str] = pupil_data.get('eyelid_landmarks')
                    except (ValueError, IndexError):
                        pass
        except zmq.Again:
            pass

        if self.enabled:
            self.update_state_machine()

    def update_state_machine(self):
        now = time.time()
        
        # Calculate combined aperture score from both eyes
        aperture0 = self._calculate_eye_aperture(self.eye_landmarks['eye0'])
        aperture1 = self._calculate_eye_aperture(self.eye_landmarks['eye1'])
        combined_aperture = aperture0 + aperture1

        # --- CALIBRATION STATES ---
        if self.state == 'CALIBRATING_OPEN':
            self.calib_samples.append(combined_aperture)
            if now - self.calib_start_time >= self.calib_duration_sec:
                self.avg_open_score = np.mean(self.calib_samples) if self.calib_samples else 0
                self.state = 'CALIBRATING_CLOSED'
                self.calib_start_time = now
                self.calib_samples = []
                self.calib_status_text.text = "CALIBRATING: Now, please CLOSE your eyes."
                logger.info(f"Eyes open calibrated. Score: {self.avg_open_score:.2f}")

        elif self.state == 'CALIBRATING_CLOSED':
            self.calib_samples.append(combined_aperture)
            if now - self.calib_start_time >= self.calib_duration_sec:
                self.avg_closed_score = np.mean(self.calib_samples) if self.calib_samples else 0
                
                # Set threshold to 30% of the way from closed to open
                self.calibrated_threshold = self.avg_closed_score + (self.avg_open_score - self.avg_closed_score) * 0.3
                self.state = 'IDLE'
                self.calib_status_text.text = f"Status: Calibrated! (Threshold: {self.calibrated_threshold:.2f})"
                logger.info(f"Eyes closed calibrated. Score: {self.avg_closed_score:.2f}")
                logger.info(f"New blink threshold set to: {self.calibrated_threshold:.2f}")

        # --- GESTURE DETECTION STATES ---
        elif self.state == 'IDLE':
            if self.calibrated_threshold > 0 and combined_aperture < self.calibrated_threshold:
                self.state = 'COLLECTING'
                self.start_time = now
                for eye_id in self.buffer: self.buffer[eye_id].clear()
                logger.info("Gesture start detected (aperture). Collecting data...")

        elif self.state == 'COLLECTING':
            # Buffer landmarks while collecting
            for eye_id in self.buffer:
                if self.eye_landmarks[eye_id] is not None:
                    self.buffer[eye_id].append({'landmarks': self.eye_landmarks[eye_id], 'timestamp': now})

            if combined_aperture >= self.calibrated_threshold:
                duration = (now - self.start_time) * 1000
                logger.info(f"Eye opening detected. Duration: {duration:.0f} ms")

                # if duration < self.min_duration_ms:
                    # self.state = 'IDLE'
                    # logger.info("Gesture too short. Resetting.")
                # else:
                    # self.process_gesture()
                    # self.state = 'COOLDOWN'
                    # self.cooldown_end_time = now + (self.cooldown_ms / 1000.0)
                    
        elif self.state == 'COOLDOWN':
            if now >= self.cooldown_end_time:
                self.state = 'IDLE'
                logger.info("Cooldown finished. Ready for next gesture.")
    
    def process_gesture(self):
        logger.info("Processing valid gesture...")
        eye0_samples = self._sample_buffer('eye0')
        eye1_samples = self._sample_buffer('eye1')

        if not eye0_samples or not eye1_samples:
            logger.warning("Not enough data to process gesture after trimming.")
            self.last_gesture = "Sampling failed"
        elif self.predictor:
            prediction = self.predictor.predict(eye0_samples, eye1_samples)
            self.last_gesture = prediction
            logger.info(f"Gesture Predicted: {prediction}")
        else:
            self.last_gesture = "Model not loaded"
        
        if self.gesture_info_text:
            self.gesture_info_text.text = f"Last Gesture: {self.last_gesture}"

    def _sample_buffer(self, eye_id):
        buffered_data = list(self.buffer[eye_id])
        if not buffered_data: return []

        start_ts = buffered_data[0]['timestamp']
        end_ts = buffered_data[-1]['timestamp']
        trim_sec = self.trim_ms / 1000.0

        trimmed_data = [d for d in buffered_data if start_ts + trim_sec <= d['timestamp'] <= end_ts - trim_sec]
        if len(trimmed_data) < self.num_samples: return []
        
        indices = np.linspace(0, len(trimmed_data) - 1, self.num_samples, dtype=int)
        return [trimmed_data[i]['landmarks'] for i in indices]

    def deinit_ui(self):
        self.remove_menu()

    def cleanup(self):
        self.pupil_sub.close()
        self.context.term()
        super().cleanup()

