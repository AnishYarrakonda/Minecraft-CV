from tests.conftest import make_screen_landmarks
from minecraft_cv.gestures.curl import CurlComboState

lm2 = make_screen_landmarks(distances={"pinky": 0.01})
cc = CurlComboState("ring", curl_fingers=("pinky",), open_fingers=("thumb", "index", "middle"))
print("Ext ring:", cc._ext(lm2, "ring"))
print("Ext pinky:", cc._ext(lm2, "pinky"))

