import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr
import os

model_path = r"D:\Fake_news_detection_BERT\fake_news_BERT_model\fake_news_BERT_model"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# Label mapping for binary model
label_map = {0: "🛑 Fake News", 1: "✅ True News"}

# Prediction function with safe try/except
def predict(text):
    try:
        if not text.strip():
            return "No input", "0%"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            label_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, label_idx].item()
            prediction = label_map.get(label_idx, f"Class {label_idx}")

        return prediction, f"{confidence*100:.2f}%"

    except Exception as e:
        print("ERROR in predict():", e)
        return "Error", "Error"

# Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=8, placeholder="Enter a news article here..."),
    outputs=[gr.Textbox(label="Prediction"), gr.Textbox(label="Confidence")],
    title="Fake News Detection using BERT",
    description="Classifies news text as Fake or True using a fine-tuned BERT model."
)

if __name__ == "__main__":
    iface.launch()
