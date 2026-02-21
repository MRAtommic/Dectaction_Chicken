import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. โหลดโมเดล (ใช้ Cache เพื่อไม่ให้โหลดซ้ำ) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path) if os.path.exists(path) else None

model = load_yolo_model(model_path)

# --- 2. ส่วนควบคุมความเร็ว ---
st.sidebar.header("🚀 Speed Optimization")
# แนะนำให้ตั้งไว้ที่ 10-15 เพื่อให้วิดีโอวิ่งไปข้างหน้าได้เร็ว
skip_frames = st.sidebar.slider("Skip Frames (ยิ่งเยอะยิ่งลื่น)", 1, 20, 10)
# ย่อขนาดภาพที่จะส่งให้ AI (ยิ่งเล็กยิ่งเร็ว)
img_size = st.sidebar.select_slider("AI Resolution", options=[160, 320, 480, 640], value=320)

uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    human_stat = st.empty()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- ประมวลผลเฉพาะเฟรมที่กำหนด ---
        if frame_count % skip_frames == 0:
            # ใช้การตรวจจับแบบรวดเร็ว (Stream Mode)
            results = model.predict(
                frame, 
                conf=0.15, 
                imgsz=img_size, # ลดขนาดรูปที่ AI ใช้ประมวลผล
                classes=[0], 
                half=True,      # ใช้โหมด Half precision (ถ้า CPU รองรับจะเร็วขึ้น)
                verbose=False
            )
            
            # วาดผลลัพธ์
            annotated_frame = results[0].plot()
            
            # อัปเดตหน้าจอ
            human_count = len(results[0].boxes)
            human_stat.metric("Detected Humans", f"{human_count}")
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1

    cap.release()
    os.remove(tfile.name)
