import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Human Detector", layout="wide")
st.title("👤 AI Human Detection System")

# --- 2. โหลดโมเดล (ใช้ชื่อไฟล์ตรงๆ) ---
model_path = "best.pt" 

@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_yolo_model(model_path)

# --- 3. ส่วนอัปโหลดวิดีโอ ---
uploaded_video = st.sidebar.file_uploader("อัปโหลดวิดีโอทดสอบ", type=['mp4', 'avi', 'mov'])
conf_threshold = st.sidebar.slider("Confidence", 0.01, 1.0, 0.15)

if uploaded_video is not None:
    # สร้างไฟล์ชั่วคราวบน Server
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ตรวจจับเฉพาะ Class 0 (Human)
        results = model.predict(
            frame, 
            conf=conf_threshold, 
            classes=[0], 
            verbose=False
        )
        
        # วาดผลลัพธ์
        annotated_frame = results[0].plot()

        # แสดงผล (ลดขนาดภาพลงเล็กน้อยเพื่อให้รันบน Cloud ได้ลื่นขึ้น)
        st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    os.remove(tfile.name) # ลบไฟล์ขั่วคราวเพื่อคืนพื้นที่ให้ Server
