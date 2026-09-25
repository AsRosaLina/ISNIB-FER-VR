import urllib.request
import os

print("Downloading face landmarker model")

model_path = 'face_landmarker.task'

if not os.path.exists(model_path):
    try:
        model_url = 'https://storage.googleapis.com/mediapipe-models/vision/face_landmarker/float16/1/face_landmarker.task'
        print(f"Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"Error downloading: {e}")
        model_path = None
else:
    print("Model already exists")

from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions
from mediapipe.tasks.python.core import base_options as base_options_module
import mediapipe as mp

class FaceAligner:
    def __init__(self, output_size=224, model_path=model_path):
        self.size = output_size
        print("Initializing FaceLandmarker...")
        
        if model_path is None:
            raise ValueError("Model file not found")
        
        # Create base options WITH model path
        base_options = base_options_module.BaseOptions(model_asset_path=model_path)
        
        # Create options
        options = FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.left_eye_idx = [33, 133]
        self.right_eye_idx = [362, 263]
        print("FaceLandmarker initialized")
    
    def _eye_center(self, landmarks, eye_idx):
        coords = np.array([[landmarks[i].x, landmarks[i].y] for i in eye_idx]).mean(axis=0)
        return coords
    
    def process(self, bgr_img):
        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = self.face_landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return cv2.resize(rgb, (self.size, self.size)).astype(np.float32)
        
        landmarks = detection_result.face_landmarks[0]
        
        left = self._eye_center(landmarks, self.left_eye_idx)
        right = self._eye_center(landmarks, self.right_eye_idx)
        
        left_px = (left[0] * w, left[1] * h)
        right_px = (right[0] * w, right[1] * h)
        
        angle = np.degrees(np.arctan2(right_px[1] - left_px[1], right_px[0] - left_px[0]))
        ctr = ((left_px[0] + right_px[0]) / 2, (left_px[1] + right_px[1]) / 2)
        
        M = cv2.getRotationMatrix2D(ctr, angle, 1.0)
        rot = cv2.warpAffine(rgb, M, (w, h))
        
        lm_array = np.array([[lm.x * w, lm.y * h] for lm in landmarks],
                           dtype=np.float32).reshape(-1, 1, 2)
        lm_rotated = cv2.transform(lm_array, M).reshape(-1, 2)
        
        x1, x2 = max(0, int(lm_rotated[:, 0].min())), min(w, int(lm_rotated[:, 0].max()))
        y1, y2 = max(0, int(lm_rotated[:, 1].min())), min(h, int(lm_rotated[:, 1].max()))
        
        px = int((x2 - x1) * 0.15)
        py = int((y2 - y1) * 0.15)
        
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        
        crop = rot[y1:y2, x1:x2]
        if crop.size == 0:
            return cv2.resize(rgb, (self.size, self.size)).astype(np.float32)
        
        return cv2.resize(crop, (self.size, self.size)).astype(np.float32)
    
    def close(self):
        if hasattr(self, 'face_landmarker'):
            self.face_landmarker = None

face_aligner = FaceAligner(output_size=224)
print("FaceAligner class defined")
