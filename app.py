import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. การตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Chicken & Human Detector", layout="wide")

# --- 2. โหลดโมเดล (ใช้ Cache เพื่อความเร็วและประหยัด RAM บน Server) ---
# ตรวจสอบว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับ app.py ใน GitHub หรือไม่
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        model = YOLO(path)
        # บน Streamlit Cloud จะไม่มี GPU (CUDA) ระบบจะเลือก CPU ให้เอง
        return model
    return None

model = load_yolo_model(model_path)

# --- 3. ส่วนประกอบหน้า Sidebar ---
st.sidebar.title("⚙️ การตั้งค่าระบบ")

if model:
    st.sidebar.success("✅ โหลดโมเดลสำเร็จ")
    # แสดงชื่อคลาสที่โมเดลตรวจจับได้
    st.sidebar.write("คลาสที่ตรวจจับได้:", list(model.names.values()))
else:
    st.sidebar.error("❌ ไม่พบไฟล์โมเดล 'best.pt' ใน Repository")
    st.stop()

# ปรับจูนความเร็ว (สำคัญมากเมื่อรันบน CPU ของ Cloud)
skip_frames = st.sidebar.slider("ข้ามเฟรม (ยิ่งเยอะยิ่งรันลื่นบนเว็บ)", 1, 30, 10)
conf_threshold = st.sidebar.slider("ค่าความมั่นใจ (Confidence)", 0.0, 1.0, 0.25)
img_size = st.sidebar.select_slider("ขนาดภาพประมวลผล", options=[160, 320, 480, 640], value=320)

uploaded_video = st.sidebar.file_uploader("อัปโหลดวิดีโอของคุณ", type=['mp4', 'avi', 'mov'])

# --- 4. ส่วนแสดงผลหลัก ---
st.title("🐔 AI Chicken & Human Detection System")
st.info("ระบบกำลังทำงานบน Cloud (CPU Mode) ความเร็วอาจช้ากว่าในเครื่องของคุณ")

if uploaded_video is not None:
    # เก็บไฟล์วิดีโอชั่วคราวเพื่ออ่านด้วย OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # เตรียมพื้นที่แสดงผล
    st_frame = st.empty()
    col1, col2 = st.columns(2)
    metric_1 = col1.empty()
    metric_2 = col2.empty()

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ประมวลผลเฉพาะเฟรมที่กำหนดเพื่อลดภาระ CPU
        if frame_count % skip_frames == 0:
            # ทำการตรวจจับ
            results = model.predict(
                frame, 
                conf=conf_threshold, 
                imgsz=img_size, 
                half=False, # บน CPU ไม่แนะนำให้ใช้ half=True
                verbose=False
            )
            
            # นับจำนวนคลาสแบบ Dynamic
            names = model.names
            detected_classes = results[0].boxes.cls.cpu().numpy()
            counts = {name: 0 for name in names.values()}
            
            for cls_id in detected_classes:
                class_name = names[int(cls_id)]
                counts[class_name] += 1
            
            # วาดผลลัพธ์
            annotated_frame = results[0].plot()
            
            # อัปเดต Metric
            class_names_list = list(counts.keys())
            if len(class_names_list) >= 1:
                metric_1.metric(f"จำนวน {class_names_list[0]}", f"{counts[class_names_list[0]]} ตัว")
            if len(class_names_list) >= 2:
                metric_2.metric(f"จำนวน {class_names_list[1]}", f"{counts[class_names_list[1]]} ตัว")
            
            # แสดงภาพ (ต้องแปลง BGR เป็น RGB สำหรับ Streamlit)
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame_rgb, use_container_width=True)
        
        frame_count += 1

    cap.release()
    os.remove(tfile.name)
    st.success("🎯 ประมวลผลวิดีโอเสร็จเรียบร้อย!")
