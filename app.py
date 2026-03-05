import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------------------
# Load Model
# -------------------

model_path = "model"   # โฟลเดอร์ที่เก็บโมเดล

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {
    0: "Negative",
    1: "Positive"
}

# -------------------
# Prediction Function
# -------------------

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_id = torch.argmax(probs).item()

    return {
        "prediction": id2label[pred_id],
        "confidence": probs[0][pred_id].item()
    }

# -------------------
# Streamlit UI
# -------------------

st.title("Product Review Sentiment Dashboard")

# เก็บรีวิว
if "reviews" not in st.session_state:
    st.session_state.reviews = []

text = st.text_area("ใส่รีวิวสินค้า")

if st.button("Analyze"):

    result = predict_sentiment(text)

    st.session_state.reviews.append({
        "text": text,
        "sentiment": result["prediction"]
    })

    st.success(f"Prediction: {result['prediction']}")

# แสดงข้อมูลทั้งหมด
df = pd.DataFrame(st.session_state.reviews)

if not df.empty:

    st.subheader("Review Data")
    st.dataframe(df)

    sentiment_count = df["sentiment"].value_counts()

    st.subheader("Sentiment Summary")

    fig, ax = plt.subplots()
    sentiment_count.plot(kind="bar", ax=ax)

    st.pyplot(fig)