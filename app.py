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
    if os.path.exists(path):
        model = YOLO(path)
        # พยายามรันบน GPU ถ้าเป็นไปได้ (สำหรับรัน Local)
        try:
            model.to('cuda')
        except:
            pass
        return model
    return None

model = load_yolo_model(model_path)

# --- 2. ส่วนควบคุมความเร็ว ---
st.sidebar.header("🚀 Speed Optimization")
skip_frames = st.sidebar.slider("Skip Frames (ยิ่งเยอะยิ่งลื่น)", 1, 20, 5)
img_size = st.sidebar.select_slider("AI Resolution", options=[160, 320, 480, 640], value=320)

uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    st_frame = st.empty()
    
    # สร้างคอลัมน์สำหรับแสดงสถิติแยกกัน
    col1, col2 = st.columns(2)
    human_stat = col1.empty()
    chicken_stat = col2.empty()
    
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- ประมวลผลเฉพาะเฟรมที่กำหนด ---
        if frame_count % skip_frames == 0:
            # ใช้การตรวจจับทั้งคลาส 0 (คน) และ 1 (ไก่)
            results = model.predict(
                frame, 
                conf=0.15, 
                imgsz=img_size, 
                classes=[0, 1], # ระบุคลาสที่ต้องการ (0=human, 1=chicken)
                half=True,      
                verbose=False
            )
            
            # ดึงข้อมูล Class IDs ที่ตรวจพบ
            detected_classes = results[0].boxes.cls.cpu().numpy()
            
            # นับจำนวนแต่ละคลาส
            num_humans = np.count_nonzero(detected_classes == 1)
            num_chickens = np.count_nonzero(detected_classes == 0)
            
            # วาดผลลัพธ์ลงบนภาพ
            annotated_frame = results[0].plot()
            
            # อัปเดตหน้าจอ (Metric)
            
            chicken_stat.metric("จำนวนไก่", f"{num_chickens} ตัว")
            human_stat.metric("จำนวนคน", f"{num_humans} ราย")
            
            # แสดงวิดีโอ
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1

    cap.release()
    os.remove(tfile.name)
    st.success("ประมวลผลวิดีโอเสร็จสิ้น")



