"""
Pupil Capture World Process Plugin for Eye Gesture Estimation.

This plugin receives eyelid landmarks and pupil confidence from eye processes,
detects eye closure gestures, and runs an inference model to predict the gesture.
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

class M2EyeGesturePredictor:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.package = None
        self.load_model()

    def load_model(self):
        """Loads the gesture prediction model."""
        if not self.model_path or not os.path.exists(self.model_path):
            logger.warning(f"Gesture model path is invalid or file not found: {self.model_path}")
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
        """Extracts delta features from a sequence of landmarks."""
        if len(landmarks_sequence) != 5:
            return None
        
        landmarks_array = np.array(landmarks_sequence)
        deltas = np.diff(landmarks_array, axis=0)
        cumulative_delta = np.cumsum(deltas, axis=0)
        
        norm_delta = np.linalg.norm(deltas) + 1e-6
        norm_cumulative = np.linalg.norm(cumulative_delta) + 1e-6
        
        normalized_deltas = deltas / norm_delta
        normalized_cumulative = cumulative_delta / norm_cumulative
        
        weighted_deltas = normalized_deltas * self.landmark_weights
        weighted_cumulative = cumulative_delta * self.landmark_weights
        
        return np.concatenate([
            weighted_deltas.flatten(),
            weighted_cumulative.flatten()
        ])

    def predict(self, eye0_landmarks, eye1_landmarks):
        """Predicts the gesture from the collected landmark data."""
        if not self.package:
            return "Model not loaded"
        try:
            features0 = self.extract_delta_features(eye0_landmarks)
            features1 = self.extract_delta_features(eye1_landmarks)
            
            if features0 is None or features1 is None:
                return "Invalid data"

            # 1. 기본 256개 피처 생성
            fusion_features = np.concatenate([features0, features1])

            # 2. <<< 핵심 수정: 3개의 추가 피처 생성 >>>
            # m2_dataset_evaluator.py의 'extract_multi_camera_features' 로직을 여기에 적용
            feature_magnitudes = [np.linalg.norm(features0), np.linalg.norm(features1)]
            max_mag, min_mag = max(feature_magnitudes), min(feature_magnitudes)
            
            # Feature 1: Magnitude Ratio
            magnitude_ratio = max_mag / (min_mag + 1e-6)
            
            # Feature 2 & 3: Consistency & Conflict (실시간 추론용으로 단순화)
            # 실제 제스처를 모르므로 방향 일관성은 1.0, 충돌 점수는 0.0으로 가정하는 것이 일반적
            direction_consistency = 1.0 
            conflict_score = np.std(feature_magnitudes) / (np.mean(feature_magnitudes) + 1e-6)

            # 3. 모든 피처를 합쳐 최종 259개 피처 생성
            enhanced_features = np.concatenate([
                fusion_features,
                [direction_consistency, magnitude_ratio, conflict_score]
            ]).reshape(1, -1)

            # 4. 스케일링 및 예측
            features_scaled = self.scaler.transform(enhanced_features)
            prediction = self.classifier.predict(features_scaled)[0]
            
            if prediction == 'left': return 'right'
            if prediction == 'right': return 'left'
            return prediction
        except Exception as e:
            logger.error(f"Gesture prediction failed: {e}")
            # 오류 메시지에 피처 개수 포함
            if 'enhanced_features' in locals():
                logger.error(f"Feature dimension mismatch. Generated: {enhanced_features.shape[1]}, Expected by scaler: {self.scaler.n_features_in_}")
            return "Prediction Error"


class Gesture_Estimator(Plugin):
    uniqueness = "by_class"
    order = 0.6

    def __init__(self, g_pool):
        super().__init__(g_pool)
        # --- User-tunable settings ---
        self.enabled = False
        self.model_path = ""
        self.confidence_threshold = 0.5
        self.min_duration_ms = 400
        self.trim_ms = 100
        self.num_samples = 5
        self.cooldown_ms = 500
        # -----------------------------

        self.state = 'IDLE'
        self.buffer = {'eye0': deque(), 'eye1': deque()}
        self.eye_confidence = {'eye0': 1.0, 'eye1': 1.0}
        self.start_time = 0.0
        self.cooldown_end_time = 0.0
        self.last_gesture = "None"
        self.gesture_info_text = None 
        self.predictor = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # ZMQ setup to receive data from eye processes
        self.context = zmq.Context()
        self.pupil_sub = self.context.socket(zmq.SUB)
        # *** IMPORTANT CHANGE: Connect to the IPC SUB URL, not PUB ***
        self.pupil_sub.connect(self.g_pool.ipc_sub_url)
        self.pupil_sub.setsockopt_string(zmq.SUBSCRIBE, "pupil.") # Subscribing to the correct topic

    def init_ui(self):
        self.add_menu()
        self.menu.label = "Gesture Estimator"
        self.menu.append(ui.Switch('enabled', self, label="Enable Gesture Estimation"))
        self.menu.append(ui.Text_Input('model_path', self, label="Gesture Model Path (.pkl)"))
        self.menu.append(ui.Button("Load Model", self.load_model))
        self.menu.append(ui.Slider('confidence_threshold', self, min=0.1, max=1.5, step=0.05, label="Confidence Threshold"))
        self.menu.append(ui.Slider('min_duration_ms', self, min=200, max=1000, step=10, label="Min Duration (ms)"))
        self.menu.append(ui.Slider('trim_ms', self, min=0, max=200, step=10, label="Trim Duration (ms)"))
        self.gesture_info_text = ui.Info_Text(f"Last Gesture: {self.last_gesture}")
        self.menu.append(self.gesture_info_text)
    
    def load_model(self):
        self.predictor = M2EyeGesturePredictor(self.model_path, self.device)

    def recent_events(self, events):
        if not self.enabled:
            return
        
        try:
            while True:
                topic, payload = self.pupil_sub.recv_multipart(flags=zmq.NOBLOCK)
                pupil_data = msgpack_unpackb_unicode(payload)
                
                topic_str = topic.decode()
                # The topic from the SUB socket will be like 'pupil.0' or 'pupil.1'
                if topic_str.startswith('pupil.'):
                    try:
                        # ▼▼▼ 수정된 부분 ▼▼▼
                        # 토픽의 두 번째 조각을 eye_id로 사용합니다.
                        parts = topic_str.split('.')
                        if len(parts) > 1 and parts[1].isdigit():
                            eye_id = int(parts[1])
                            eye_id_str = f"eye{eye_id}"

                            if eye_id_str in self.eye_confidence:
                                self.eye_confidence[eye_id_str] = pupil_data.get('confidence', 0.0)
                                
                                if self.state == 'COLLECTING':
                                    landmarks = pupil_data.get('eyelid_landmarks')
                                    if landmarks is not None:
                                        timestamp = pupil_data.get('timestamp')
                                        self.buffer[eye_id_str].append({'landmarks': landmarks, 'timestamp': timestamp})
                        # ▲▲▲
                    except (ValueError, IndexError):
                        # eye_id를 파싱할 수 없는 다른 'pupil.' 토픽은 조용히 무시합니다.
                        pass
        except zmq.Again:
            pass

        self.update_state_machine()

    def update_state_machine(self):
        now = time.time()
        combined_confidence = sum(self.eye_confidence.values())

        if self.state == 'IDLE':
            if combined_confidence < self.confidence_threshold:
                self.state = 'COLLECTING'
                self.start_time = now
                for eye_id in self.buffer: self.buffer[eye_id].clear()
                logger.info("Gesture start detected. Collecting data...")

        elif self.state == 'COLLECTING':
            if combined_confidence >= self.confidence_threshold:
                duration = (now - self.start_time) * 1000
                logger.info(f"Eye opening detected. Duration: {duration:.0f} ms")

                if duration < self.min_duration_ms:
                    self.state = 'IDLE'
                    logger.info("Gesture too short (blink). Resetting.")
                else:
                    self.process_gesture()
                    self.state = 'COOLDOWN'
                    self.cooldown_end_time = now + (self.cooldown_ms / 1000.0)
                    
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
            return
            
        if self.predictor:
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

