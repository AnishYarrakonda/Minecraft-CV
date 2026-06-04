import numpy as np
import pytest

from minecraft_cv.joystick.wrist_tilt import WristTiltJoystick, wrist_tilt_vector

def test_wrist_tilt_vector() -> None:
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0, :2] = [0.5, 0.5]
    landmarks[9, :2] = [0.4, 0.3]
    
    vec = wrist_tilt_vector(landmarks)
    assert np.allclose(vec, [-0.1, -0.2])

def test_wrist_tilt_joystick_deadzone() -> None:
    joy = WristTiltJoystick(deadzone_deg=0.1, sensitivity=1.0, smoothing=0.0)
    
    # Neutral vector
    joy.update(np.array([0.0, -0.2]))
    assert np.allclose(joy.neutral, [0.0, -0.2])
    
    # Inside deadzone
    out1 = joy.update(np.array([0.05, -0.2]))
    assert np.linalg.norm(out1) == 0.0
    
    # Outside deadzone
    out2 = joy.update(np.array([0.2, -0.2]))
    assert np.linalg.norm(out2) > 0.0

def test_wrist_tilt_joystick_direction() -> None:
    joy = WristTiltJoystick(deadzone_deg=0.05, sensitivity=1.0, smoothing=0.0)
    joy.update(np.array([0.0, -0.2]))  # Neutral
    
    # Tilt right (+x)
    out = joy.update(np.array([0.2, -0.2]))
    assert out[0] > 0.0
    assert out[1] == 0.0
    
    # Tilt up/forward (-y)
    out = joy.update(np.array([0.0, -0.4]))
    assert out[0] == 0.0
    assert out[1] < 0.0
    
    # Tilt down/back (+y)
    out = joy.update(np.array([0.0, -0.1]))
    assert out[0] == 0.0
    assert out[1] > 0.0

def test_wrist_tilt_joystick_recenter() -> None:
    joy = WristTiltJoystick(deadzone_deg=0.05, sensitivity=1.0, smoothing=0.0)
    joy.update(np.array([0.0, -0.2]))
    
    joy.recenter_at(np.array([0.5, -0.5]))
    assert np.allclose(joy.neutral, [0.5, -0.5])
    
    out = joy.update(np.array([0.5, -0.5]))
    assert np.linalg.norm(out) == 0.0
