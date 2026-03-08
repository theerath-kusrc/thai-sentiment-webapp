import gradio as gr
import pandas as pd
from transformers import pipeline

# 1. 📍 ชื่อโมเดลของเพื่อนที่เป็นเวอร์ชัน 2-Class
MODEL_ID = 'Kanyasiri/wangchanberta-wongnai-sentiment'

# 2. ปรับ Demo ให้เป็นประโยคที่เน้น บวก หรือ ลบ ชัดเจน
DEMOS = [
    ["ร้านนี้บริการดีมาก อาหารอร่อยทุกอย่าง ประทับใจสุดๆ"],
    ["แย่มากครับ รออาหารนานเป็นชั่วโมง แถมพนักงานพูดจาไม่ดี"],
    ["รสชาติอาหารถือว่าใช้ได้เลย ราคาไม่แพง คุ้มค่าเงิน"],
    ["ผิดหวังมาก อาหารไม่สด สกปรก ไม่กล้ากินต่อเลย"],
    ["เป็นร้านโปรดเลย มาประจำ อาหารคุณภาพดีสม่ำเสมอ"]
]

# 3. โหลดโมเดล
try:
    # กำหนด top_k=None เพื่อให้แสดงผล Bar ทั้ง 2 คลาส
    classifier = pipeline('text-classification', model=MODEL_ID, top_k=None)
except Exception as e:
    print(f"Error loading model: {e}")
    # Mockup สำหรับทดสอบ UI
    def classifier(text):
        return [[{'label': 'LABEL_0', 'score': 0.9}, {'label': 'LABEL_1', 'score': 0.1}]]

def process_prediction(text):
    if not text.strip():
        return {"กรุณาพิมพ์ข้อความ": 1.0}, "-"
        
    results = classifier(text)[0]
    
    # 🚨 สำคัญ: Mapping ให้ตรงกับที่เพื่อนเทรนมา (0=Pos, 1=Neg)
    label_map = {'LABEL_0': 'Positive 😊', 'LABEL_1': 'Negative 😠'}
    
    output_dict = {}
    for res in results:
        label_name = label_map.get(res['label'], res['label'])
        output_dict[label_name] = res['score']
    
    best_label = max(output_dict, key=output_dict.get)
    return output_dict, best_label

# 4. ฟังก์ชัน Single & History
def predict_single(text, history):
    scores_dict, best_label = process_prediction(text)
    confidence = scores_dict.get(best_label, 0)
    new_entry = [text, best_label, f"{confidence:.2%}"]
    updated_history = [new_entry] + history
    return scores_dict, updated_history

# 5. ฟังก์ชัน Batch CSV
def predict_batch(file):
    if file is None: return None
    df = pd.read_csv(file.name)
    review_col = 'review' if 'review' in df.columns else df.columns[0]
    sentiments, confs = [], []
    for text in df[review_col].astype(str):
        scores_dict, best_label = process_prediction(text)
        sentiments.append(best_label)
        confs.append(f"{scores_dict.get(best_label, 0):.2%}")
    df['Sentiment'] = sentiments
    df['Confidence %'] = confs
    return df

# 6. UI Setup
with gr.Blocks(theme=gr.themes.Soft(), title='Wongnai Sentiment (2-Class)') as demo:
    gr.Markdown("# 🇹🇭 Thai Sentiment Analysis (2-Class)")
    gr.Markdown("วิเคราะห์รีวิว Wongnai: **Positive** หรือ **Negative** เท่านั้น (แม่นยำสูงกว่า)")
    
    with gr.Tabs():
        with gr.Tab("วิเคราะห์ข้อความ"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Textbox(label="รีวิวภาษาไทย", lines=3)
                    btn = gr.Button("🔍 วิเคราะห์", variant='primary')
                    gr.Examples(examples=DEMOS, inputs=inp)
                with gr.Column():
                    out_label = gr.Label(label='Sentiment Score', num_top_classes=2)
            
            gr.Markdown("### 🕒 Query History")
            history_state = gr.State([])
            history_table = gr.Dataframe(headers=["รีวิว", "ผลลัพธ์", "ความมั่นใจ %"])
            btn.click(fn=predict_single, inputs=[inp, history_state], outputs=[out_label, history_state])
            history_state.change(fn=lambda h: h, inputs=history_state, outputs=history_table)

        with gr.Tab("Batch CSV"):
            csv_inp = gr.File(label="Upload CSV", file_types=['.csv'])
            csv_btn = gr.Button("📂 วิเคราะห์แบบกลุ่ม")
            csv_out = gr.Dataframe()
            csv_btn.click(fn=predict_batch, inputs=csv_inp, outputs=csv_out)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
    

