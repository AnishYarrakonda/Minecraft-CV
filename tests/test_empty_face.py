from minecraft_cv.config import Settings
from minecraft_cv.gestures.face_gestures import FaceGestureStateMachine
from minecraft_cv.tracking.face_tracker import FaceResult

settings = Settings().gestures.face
sm = FaceGestureStateMachine(settings)
print(sm.status())
res = sm.update(FaceResult())
print([d.name for d in sm._detectors if d._is_active])
