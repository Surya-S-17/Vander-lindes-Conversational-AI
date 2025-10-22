from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import joblib

model_path = "intent_classifier_pytorch"  
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
le = joblib.load("label_encoder.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
@app.post("/predict-intent/")
async def predict_intent(request: QueryRequest):
    query = request.query.lower()
    
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good evening"]
    
    if any(word in query for word in greeting_keywords):
        predicted_intent = "greeting"
    else:
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_id = torch.argmax(outputs.logits, dim=-1).item()
            predicted_intent = le.inverse_transform([predicted_class_id])[0]

    return {"intent": predicted_intent, "query": query}
