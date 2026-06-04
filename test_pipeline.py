import cv2, numpy as np
from minecraft_cv.pipeline import Pipeline
from minecraft_cv.config import Settings
from minecraft_cv.tracking.face_tracker import FaceTracker

settings = Settings()
face_tracker = FaceTracker()
p = Pipeline.from_settings(settings)

frame = np.zeros((480, 640, 3), dtype=np.uint8)
res = face_tracker.detect(frame, 0)
print("FaceResult:", res.blendshapes)
step = p.step([], res)
print("status:", step.face_status)
print("face_gestures:", step.face_gestures)
