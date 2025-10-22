from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib

# -------------------- Load model, tokenizer, and label encoder --------------------
model_path = "intent_classifier_pytorch"  # Path to saved model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
le = joblib.load("label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------- Define FastAPI app --------------------
app = FastAPI()

# Define the request body for input query
class QueryRequest(BaseModel):
    query: str

# -------------------- Define the intent prediction endpoint --------------------
@app.post("/predict-intent/")
async def predict_intent(request: QueryRequest):
    query = request.query.lower()
    
    # -------------------- Greeting keywords --------------------
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
    
    # 1️⃣ Check if greeting
    if any(word in query for word in greeting_keywords):
        predicted_intent = "greeting"
    else:
        # 2️⃣ Predict intent using model
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
            predicted_intent = le.inverse_transform([predicted_class_id])[0]

    # 3️⃣ Return structured output
    return {"intent": predicted_intent, "query": query}

# -------------------- Run the FastAPI server --------------------
# If using Uvicorn to serve this app, run the following in your terminal:
# uvicorn main:app --reload

