# 📰 Fake News Detection using BERT

This project uses a **BERT-based model** to classify news articles as **Fake** or **Real**.  
The model is fine-tuned on the [Kaggle Fake News dataset]https://www.kaggle.com/datasets and deployed with a **Gradio interface** for easy interaction.

---

🖼️ Demo app linkv: [https://huggingface.co/spaces/AKALYAS/Fake_news_detectio_BERT](https://huggingface.co/spaces/AKALYAS/CNN_Task?logs=container)

---

## 🚀 Features
- Uses **BERT (Bidirectional Encoder Representations from Transformers)** for text classification  
- Detects whether a given news text is **Fake** or **Real**  
- Simple and interactive **Gradio Web App**  
- Ready for deployment on **Hugging Face Spaces** or local machine  

---

## 📂 Project Structure

├── app.py # Main Gradio app

├── requirements.txt # Dependencies

├── archive (2)/ # Dataset folder

│ ├── Fake.csv

│ ├── True.csv

├── results/ # Fine-tuned BERT model (config.json, pytorch_model.bin, tokenizer.json etc.)

└── README.md # Project documentation



---

## ⚙️ Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/fake-news-detection-bert.git
   cd fake-news-detection-bert

   
2.Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


3.Install dependencies:

pip install -r requirements.txt

---

▶️ Usage
1. Run Locally
python app.py


Then open the Gradio link in your browser (default: http://127.0.0.1:7860).

2. Deploy on Hugging Face Spaces

Push your code (including app.py, requirements.txt, results/) to a new Space.

Choose Gradio SDK when creating the Space.

The app will build automatically and launch in your Space.

---

📊 Dataset

We used the Fake and Real News Dataset from Kaggle:

Fake.csv → Fake news articles

True.csv → Real news articles

Preprocessed and labeled (0 = Fake, 1 = Real)

---

🧠 Model

Base model: bert-base-uncased

Fine-tuned using transformers on the news dataset

Outputs logits for two classes: Fake (0) and Real (1)

---



📦 Requirements

See requirements.txt
. Example:

torch==2.1.0
transformers==4.36.2
datasets==2.19.0
gradio==5.8.0

---

✨ Future Work

Improve accuracy with larger transformer models (RoBERTa, DistilBERT)

Add confidence scores with probabilities

Deploy as a REST API using FastAPI

---

🖼️ Demo app :

<img width="1867" height="527" alt="Screenshot 2025-10-03 220721" src="https://github.com/user-attachments/assets/3feee073-b3db-4633-afaf-8550f4c642e4" />

----
