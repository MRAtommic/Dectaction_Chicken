import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
import time # เพิ่ม time เข้ามาช่วยจัดการจังหวะ

st.set_page_config(page_title="AI Human Detector", layout="wide")
st.title("👤 AI Human Detection (Smooth View)")

# --- โหลดโมเดล ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best.pt")

@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_yolo_model(model_path)

# --- ส่วนควบคุม ---
uploaded_video = st.sidebar.file_uploader("อัปโหลดวิดีโอ", type=['mp4', 'avi', 'mov'])
conf_threshold = st.sidebar.slider("Confidence", 0.01, 1.0, 0.20)
# เพิ่ม Skip Frames ให้มากขึ้นเพื่อลดอาการค้าง
skip_frames = st.sidebar.select_slider("ปรับความลื่น (ข้ามเฟรม)", options=[1, 3, 5, 10], value=5)

if uploaded_video is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # สร้างพื้นที่แสดงผลแบบ Container เพื่อให้ภาพกับตัวเลขไปด้วยกัน
    display_col, stat_col = st.columns([3, 1])
    st_frame = display_col.empty() 
    human_stat = stat_col.empty()
    
    frame_count = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            # 1. ย่อภาพให้เล็กลงมากที่สุดที่ AI ยังมองเห็น (ช่วยให้เร็วขึ้นมหาศาล)
            display_frame = cv2.resize(frame, (480, 270)) 
            
            # 2. ให้ AI ทำงาน
            results = model.predict(
                display_frame, 
                conf=conf_threshold, 
                classes=[0], 
                verbose=False
            )
            
            # 3. เตรียมภาพและตัวเลขให้เสร็จก่อนโชว์
            annotated_frame = results[0].plot(line_width=2)
            class_ids = results[0].boxes.cls.cpu().numpy()
            human_count = np.count_nonzero(class_ids == 0)

            # 4. แสดงผลพร้อมกัน
            human_stat.metric("จำนวนคน", f"{human_count} ราย")
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # 5. ใส่ delay สั้นๆ เพื่อให้ Browser มีเวลา Render ภาพ
            time.sleep(0.01) 
        
        frame_count += 1

    cap.release()
    os.remove(tfile.name)
