import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บให้รองรับจอภาพขนาดกว้าง ---
st.set_page_config(page_title="AI Chicken & Human Detector", layout="wide")
st.title("🐔 & 👤 AI Video Detection System (Mannequin Edition)")
st.write("ระบบตรวจจับไก่และคน")

# --- 2. โหลดโมเดล (ใช้ Path แบบ Relative เพราะไฟล์อยู่ข้างกัน) ---
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

# --- 3. ส่วนควบคุมด้านข้าง (Sidebar) ---
if model is None:
    st.sidebar.error(f"❌ ไม่พบไฟล์ {model_path} กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์เดียวกันกับ app.py")
    st.stop()
else:
    st.sidebar.success("✅ โหลดโมเดลเรียบร้อย!")

st.sidebar.divider()
uploaded_video = st.sidebar.file_uploader("เลือกไฟล์วิดีโอทดสอบ", type=['mp4', 'avi', 'mov'])

# ปรับ Conf เริ่มต้นไว้ที่ 0.10 เพื่อให้ AI 'กล้า' ทายหุ่นจำลองมากขึ้น
conf_threshold = st.sidebar.slider("ความเชื่อมั่น (Confidence)", 0.05, 1.0, 0.10)
iou_threshold = st.sidebar.slider("การซ้อนทับ (IOU Threshold)", 0.1, 1.0, 0.45)

# --- 4. การประมวลผลวิดีโอ ---
if uploaded_video is not None:
    # สร้างไฟล์ชั่วคราวเพื่ออ่านวิดีโอ
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() 
    
    # แสดงตัวเลขสถิติด้านบน
    col1, col2 = st.columns(2)
    chicken_stat = col1.empty()
    human_stat = col2.empty()

    btn_stop = st.sidebar.button("หยุดการประมวลผล", type="primary")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or btn_stop:
            break


            # แก้ไขในไฟล์ app.py ส่วน model.predict
        results = model.predict(
        frame, 
        conf=0.01,          # ลดเหลือ 0.01 (ถ้ามีเงาคล้ายคนนิดเดียวให้แสดงเลย)
        iou=0.45, 
        classes=[0, 1],    
        agnostic_nms=True, 
        verbose=False
    )
        
        # นับจำนวนที่พบในเฟรมนี้
        class_ids = results[0].boxes.cls.cpu().numpy()
        chicken_count = np.count_nonzero(class_ids == 0)
        human_count = np.count_nonzero(class_ids == 1)

        # อัปเดตตัวเลขแสดงผล
        chicken_stat.metric("จำนวนไก่ที่พบ", f"{chicken_count} ตัว")
        human_stat.metric("จำนวนคน/หุ่นที่พบ", f"{human_count} ราย")

        # วาดกรอบและ Label (ปรับขนาดเส้นและฟอนต์ให้ชัดเจน)
        annotated_frame = results[0].plot(line_width=2, font_size=1.0)

        # แสดงผลวิดีโอ (แปลงสีจาก BGR เป็น RGB อัตโนมัติผ่าน st.image)
        st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

    cap.release()
    st.success("การทดสอบเสร็จสิ้น")
    os.remove(tfile.name) # ลบไฟล์ชั่วคราว
else:
    st.info("👈 กรุณาอัปโหลดวิดีโอที่แถบด้านข้างเพื่อเริ่มการตรวจจับ")