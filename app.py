import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------

model_path = "model" 

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

id2label = {
    0: "Negative",
    1: "Positive"
}

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

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

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("Product Review Sentiment Dashboard")


if "reviews" not in st.session_state:
    st.session_state.reviews = []


st.subheader("Upload CSV for Batch Analysis")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Preview Data")
    st.dataframe(data.head())

    if st.button("Analyze CSV"):

        results = []

        for text in data["text"]:
            result = predict_sentiment(text)
            results.append(result["prediction"])

        data["sentiment"] = results

        st.write("Analysis Result")
        st.dataframe(data)

        st.metric("Total Reviews", len(data))

        positive_ratio = (data["sentiment"] == "Positive").mean() * 100
        st.metric("Positive Rate", f"{positive_ratio:.1f}%")

        sentiment_count = data["sentiment"].value_counts()

        fig, ax = plt.subplots()

        ax.pie(
            sentiment_count,
            labels=sentiment_count.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Sentiment Distribution")

        st.pyplot(fig)