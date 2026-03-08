import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# 1. ⚙️ การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Thai Sentiment Analysis", page_icon="🇹🇭", layout="wide")

# 2. 📍 เชื่อมต่อโมเดลของเพื่อน (Prang)
MODEL_ID = 'Kanyasiri/wangchanberta-wongnai-sentiment'

@st.cache_resource
def load_model():
    # ใช้ pipeline จาก transformers
    return pipeline('text-classification', model=MODEL_ID, top_k=None)

try:
    classifier = load_model()
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")
    classifier = None

# 3. ส่วนหัวของเว็บไซต์
st.title("🇹🇭 Thai Sentiment Analysis: Wongnai Reviews")
st.markdown("วิเคราะห์ความรู้สึกจากรีวิวร้านอาหารด้วยโมเดล **WangchanBERTa (2-Class Version)**")

# 4. สร้าง Sidebar สำหรับประวัติการใช้งาน
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🕒 ประวัติการวิเคราะห์")
    if st.session_state.history:
        for item in st.session_state.history:
            st.info(f"**ข้อความ:** {item['text'][:30]}...\n**ผลลัพธ์:** {item['label']}")
    else:
        st.write("ยังไม่มีประวัติ")

# 5. ส่วนหลักของแอป (Tabs)
tab1, tab2 = st.tabs(["🔍 วิเคราะห์ข้อความเดี่ยว", "📂 อัปโหลดไฟล์ CSV"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area("พิมพ์รีวิวที่นี่:", placeholder="ตัวอย่าง: อาหารอร่อยมาก บริการดีเยี่ยม...", height=150)
        predict_button = st.button("วิเคราะห์ความรู้สึก", type="primary")

    with col2:
        if predict_button and input_text:
            if classifier:
                results = classifier(input_text)[0]
                
                # Mapping Label ตามที่เพื่อนเทรนมา (0=Pos, 1=Neg)
                label_map = {'LABEL_0': 'Positive 😊', 'LABEL_1': 'Negative 😠'}
                
                # จัดรูปแบบผลลัพธ์เพื่อแสดงผล
                formatted_results = {label_map.get(r['label'], r['label']): r['score'] for r in results}
                best_label = max(formatted_results, key=formatted_results.get)
                
                # แสดงผลลัพธ์เป็น Progress Bar
                st.subheader("ผลการวิเคราะห์:")
                for label, score in formatted_results.items():
                    st.write(f"{label}: {score:.2%}")
                    st.progress(score)
                
                # เก็บลงประวัติ
                st.session_state.history.insert(0, {"text": input_text, "label": best_label})
            else:
                st.warning("โมเดลยังไม่พร้อมใช้งาน")

with tab2:
    st.subheader("วิเคราะห์หลายรายการผ่านไฟล์ CSV")
    uploaded_file = st.file_uploader("เลือกไฟล์ CSV (ต้องมีคอลัมน์ชื่อ 'review')", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if st.button("เริ่มวิเคราะห์ไฟล์"):
            with st.spinner('กำลังประมวลผล...'):
                review_col = 'review' if 'review' in df.columns else df.columns[0]
                
                def get_prediction(text):
                    res = classifier(str(text))[0]
                    # หาตัวที่คะแนนสูงสุด
                    top = max(res, key=lambda x: x['score'])
                    return label_map.get(top['label'], top['label']), f"{top['score']:.2%}"

                df[['Sentiment', 'Confidence']] = df[review_col].apply(lambda x: pd.Series(get_prediction(x)))
                
                st.success("วิเคราะห์เสร็จสิ้น!")
                st.dataframe(df)
                
                # ปุ่มดาวน์โหลดผลลัพธ์
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("ดาวน์โหลดผลลัพธ์ (CSV)", csv, "results.csv", "text/csv")
