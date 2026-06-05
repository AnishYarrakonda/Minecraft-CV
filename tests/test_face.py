import time

import cv2

from minecraft_cv.tracking.face_tracker import FaceTracker


def main():
    img = cv2.imread("face.jpg")
    if img is None:
        print("Error: face.jpg not found.")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Assuming models/face_landmarker.task exists in the repo
    # It might be in the parent directory if testing from worktree, so we use a fallback if not found
    import os
    model_path = "models/face_landmarker.task"
    if not os.path.exists(model_path):
        # Fallback to parent repo
        model_path = "/Users/anish_1_2_3/Documents/minecraft_cv/models/face_landmarker.task"

    tracker = FaceTracker(model_path=model_path)
    result = tracker.detect(rgb, int(time.time() * 1000))
    print(f"Face blendshapes count: {len(result.blendshapes)}")
    if result.blendshapes:
        print("Successfully extracted blendshapes from face.jpg!")
    else:
        print("Failed to extract blendshapes.")

if __name__ == "__main__":
    main()
