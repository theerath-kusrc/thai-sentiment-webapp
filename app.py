import streamlit as st
import pandas as pd
from transformers import pipeline
import torch

# 1. การตั้งค่าหน้าเว็บ (UI/UX)
st.set_page_config(
    page_title="Wongnai Sentiment Analysis (2-Class)",
    page_icon="🍽️",
    layout="wide"
)


from transformers import CamembertTokenizer, CamembertForSequenceClassification

# บังคับใช้ชื่อโมเดลที่คุณระบุ
model_name = "Kanyasiri/wangchanberta-wongnai-3class"

# โหลด Tokenizer และ Model โดยระบุ Class ให้ตรงกับสถาปัตยกรรม WangchanBERTa
# เพิ่ม trust_remote_code=True เผื่อในกรณีที่มีสคริปต์พิเศษใน Repo
tokenizer = CamembertTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = CamembertForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)

# กำหนด Label Map ให้ตรงกับข้อมูลที่คุณระบุไว้
LABEL_MAP = {0: "Positive", 1: "Neutral", 2: "Negative"}

@st.cache_resource
def load_model():
    try:
        # โหลด pipeline สำหรับ Text Classification
        return pipeline('text-classification', model=MODEL_ID, top_k=None)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

classifier = load_model()

# 3. ส่วนหัวของเว็บไซต์
st.title("🇹🇭 Thai Sentiment Analysis: Wongnai Reviews")
st.markdown("วิเคราะห์ความรู้สึกจากรีวิวด้วยโมเดล **WangchanBERTa (2-Class)** — เน้นความแม่นยำสูง (Positive / Negative)")

# 4. ระบบจัดการประวัติ (Sidebar)
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("🕒 History")
    if st.session_state.history:
        for i, item in enumerate(st.session_state.history):
            st.info(f"**{i+1}.** {item['text'][:30]}...\n**Result:** {item['label']}")
    else:
        st.write("ยังไม่มีประวัติ")
    if st.button("ล้างประวัติ"):
        st.session_state.history = []
        st.rerun()

# 5. ส่วนการทำงานหลัก (Tabs)
tab1, tab2 = st.tabs(["🔍 วิเคราะห์ข้อความเดี่ยว", "📂 อัปโหลดไฟล์ CSV"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area("พิมพ์รีวิวร้านอาหารที่นี่ (แนะนำรีวิวที่แสดงอารมณ์ชัดเจน):", 
                                 placeholder="ตัวอย่าง: อาหารอร่อยมาก ประทับใจสุดๆ...", height=150)
        predict_btn = st.button("วิเคราะห์ผล", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 💡 ตัวอย่างรีวิว (Demo)")
        examples = [
            "อร่อยมากครับ แนะนำเลยร้านนี้ บริการดีมาก",
            "รสชาติแย่มาก อาหารไม่สดเลย เสียความรู้สึกที่สุด",
            "เป็นร้านประจำเลย มาทุกครั้งก็ประทับใจทุกครั้ง"
        ]
        for ex in examples:
            if st.button(ex):
                st.info(f"คัดลอกข้อความนี้ไปวางด้านบน: {ex}")

    with col2:
        if predict_btn and input_text:
            if classifier:
                with st.spinner('กำลังประมวลผล...'):
                    results = classifier(input_text)[0]
                    
                    # 🚨 Mapping Label ตามที่เพื่อนเทรนมา (2-Class)
                    # 0: Positive 😊, 1: Negative 😠
                    label_map = {
                        'LABEL_0': 'Positive 😊', 
                        'LABEL_1': 'Negative 😠'
                    }
                    
                    scores_dict = {label_map.get(r['label'], r['label']): r['score'] for r in results}
                    best_label = max(scores_dict, key=scores_dict.get)
                    
                    st.subheader("ผลการวิเคราะห์:")
                    # แสดงผลเป็น Progress Bar แยกสี (เขียว/แดง)
                    for label, score in scores_dict.items():
                        st.write(f"**{label}**")
                        st.progress(score)
                        st.caption(f"Confidence: {score:.2%}")
                    
                    # บันทึกประวัติ
                    st.session_state.history.insert(0, {"text": input_text, "label": best_label})
            else:
                st.warning("ระบบยังโหลดโมเดลไม่สำเร็จ หรือ MODEL_ID ไม่ถูกต้อง")

with tab2:
    st.subheader("วิเคราะห์หลายรายการ (Batch Processing)")
    st.write("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ชื่อ `review` เพื่อวิเคราะห์ผลลัพธ์แบบกลุ่ม")
    csv_file = st.file_uploader("เลือกไฟล์ CSV", type=["csv"])
    
    if csv_file and classifier:
        df = pd.read_csv(csv_file)
        if st.button("เริ่มวิเคราะห์ไฟล์ CSV"):
            with st.spinner('กำลังคำนวณ...'):
                col_name = 'review' if 'review' in df.columns else df.columns[0]
                
                def get_sentiment(text):
                    res = classifier(str(text))[0]
                    top = max(res, key=lambda x: x['score'])
                    m = {'LABEL_0': 'Positive', 'LABEL_1': 'Negative'}
                    return m.get(top['label'], top['label']), f"{top['score']:.2%}"

                df[['Result', 'Confidence']] = df[col_name].apply(lambda x: pd.Series(get_sentiment(x)))
                st.success("สำเร็จ!")
                st.dataframe(df, use_container_width=True)
                
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 ดาวน์โหลดผลลัพธ์ (CSV)", csv_data, "sentiment_results.csv", "text/csv")




