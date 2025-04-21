import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from huggingface_hub import hf_hub_download

import numpy as np

# ✅ Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="NancyAthghara23/YT_Sentiment_analysis",  # ✅ Replace with your actual model repo ID
    filename="model.pth"  # ✅ Make sure this is exactly how it's named in the repo
)

# ✅ Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# ✅ Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# ✅ Streamlit UI
st.title("🎭 YouTube Comment Sentiment Analyzer")
comment = st.text_area("Enter a comment:")

if st.button("Analyze Sentiment"):
    with torch.no_grad():
        inputs = tokenizer(comment, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

        st.write(f"**Predicted Sentiment:** {label_map[pred]}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
