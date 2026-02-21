import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="AI Human Detector Cloud", layout="wide")
st.title("👤 AI Human Detection System (Cloud Optimized)")
st.write("ระบบตรวจจับคนมุมสูง (เวอร์ชันแก้ไข Path และความเร็ว)")

# --- 2. โหลดโมเดล (วิธีหา Path แบบยืดหยุ่นเพื่อให้รันบน GitHub ได้) ---
# ดึงตำแหน่งที่ตั้งของไฟล์ app.py ปัจจุบัน
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# รวมชื่อไฟล์เข้าไป เพื่อระบุตำแหน่งที่แน่นอนของ best.pt
model_path = os.path.join(BASE_DIR, "best.pt")

@st.cache_resource
def load_yolo_model(path):
    # ตรวจสอบว่าไฟล์มีอยู่จริงหรือไม่ก่อนโหลด
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            st.error(f"❌ โหลดโมเดลไม่ได้: {e}")
            return None
    else:
        # หากหาไม่เจอ จะแสดง Error พร้อมบอกตำแหน่งที่ระบบกำลังหาอยู่
        st.sidebar.error(f"❌ ไม่พบไฟล์โมเดลในตำแหน่ง: {path}")
        st.sidebar.info("กรุณาตรวจสอบว่าไฟล์ best.pt อยู่ในโฟลเดอร์เดียวกับ app.py บน GitHub หรือไม่")
        return None

model = load_yolo_model(model_path)

# --- 3. ส่วนควบคุมด้านข้าง ---
if model is not None:
    st.sidebar.success("✅ เชื่อมต่อโมเดลสำเร็จ!")

uploaded_video = st.sidebar.file_uploader("อัปโหลดวิดีโอทดสอบ", type=['mp4', 'avi', 'mov'])
conf_threshold = st.sidebar.slider("Confidence", 0.01, 1.0, 0.15)
skip_frames = st.sidebar.select_slider("ข้ามเฟรมเพื่อความลื่นไหล", options=[1, 2, 3, 5], value=2)

# --- 4. การประมวลผลวิดีโอ ---
if uploaded_video is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() 
    human_stat = st.empty()
    
    frame_count = 0 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            # ย่อขนาดเพื่อให้ CPU ของ Cloud ทำงานทัน
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

            # วาดผลลัพธ์
            annotated_frame = results[0].plot(line_width=2)
            st_frame.image(annotated_frame, channels="BGR", use_container_width=True)
        
        frame_count += 1

    cap.release()
    st.success("การประมวลผลเสร็จสิ้น")
    os.remove(tfile.name)
elif model is None:
    st.warning("⚠️ ระบบยังไม่พร้อมใช้งานเนื่องจากโหลดโมเดลไม่สำเร็จ")
else:
    st.info("👈 กรุณาอัปโหลดวิดีโอเพื่อเริ่มการทดสอบ")
