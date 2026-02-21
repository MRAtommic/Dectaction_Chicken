import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Human Detector Cloud", layout="wide")
st.title("👤 AI Human Detection System (Cloud Optimized)")
st.write("ระบบตรวจจับคนมุมสูง (เวอร์ชันปรับปรุงความเร็วสำหรับการรันบน Cloud)")

# --- 2. โหลดโมเดล ---
model_path = "best.pt" 

@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    return None

model = load_yolo_model(model_path)

# --- 3. ส่วนควบคุมด้านข้าง ---
if model is None:
    st.sidebar.error(f"❌ ไม่พบไฟล์ {model_path} กรุณาตรวจสอบใน GitHub ของคุณ")
    st.stop()

uploaded_video = st.sidebar.file_uploader("อัปโหลดวิดีโอทดสอบ", type=['mp4', 'avi', 'mov'])
conf_threshold = st.sidebar.slider("Confidence", 0.01, 1.0, 0.15)

# ตัวเลือกเพื่อเพิ่มความเร็ว (Skip Frames)
skip_frames = st.sidebar.select_slider("ข้ามเฟรมเพื่อความลื่นไหล (Skip Frames)", options=[1, 2, 3, 5], value=2)

# --- 4. การประมวลผลวิดีโอ ---
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() 
    human_stat = st.empty()
    
    frame_count = 0 # ตัวนับเฟรม

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # --- เทคนิคข้ามเฟรมเพื่อให้วิดีโอไม่ค้างบน Server ---
        if frame_count % skip_frames == 0:
            # ย่อขนาดภาพลง 50% เพื่อให้ CPU ประมวลผลไวขึ้น
            small_frame = cv2.resize(frame, (640, 360)) 
            
            results = model.predict(
                small_frame, 
                conf=conf_threshold, 
                classes=[0],       # คลาสคนตามไฟล์ data.yaml ของคุณ
                agnostic_nms=True, 
                verbose=False
            )
            
            # นับจำนวนคน
            class_ids = results[0].boxes.cls.cpu().numpy()
            human_count = np.count_nonzero(class_ids == 0)
            human_stat.metric("จำนวนคนที่ตรวจพบ", f"{human_count} ราย")

            # วาดกรอบลงบนภาพ
            annotated_frame = results[0].plot(line_width=2)

            # แสดงผล
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1

    cap.release()
    st.success("การประมวลผลเสร็จสิ้น")
    os.remove(tfile.name)
else:
    st.info("👈 อัปโหลดวิดีโอที่แถบด้านซ้ายเพื่อเริ่มการทดสอบ")
