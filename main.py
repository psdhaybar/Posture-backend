from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import tempfile
from posture_rules import analyze_squat, analyze_desk_posture

app = FastAPI()

# Allow frontend to connect (adjust this later for security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-posture")
async def analyze_posture(file: UploadFile = File(...), mode: str = "squat"):
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    bad_frames = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]

            if mode == "squat":
                bad, info = analyze_squat(landmarks)
            else:
                bad, info = analyze_desk_posture(landmarks)

            if bad:
                bad_frames.append({"frame": frame_count, **info})

        frame_count += 1

    cap.release()
    return {"bad_posture_frames": bad_frames}
