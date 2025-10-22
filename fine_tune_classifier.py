# Install the required packages

# -------------------- Imports --------------------
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import numpy as np

# -------------------- 1️⃣ Load Data --------------------
df = pd.read_csv("intents.csv")  # Columns: 'query', 'intent'

# -------------------- 2️⃣ Encode Labels --------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])

# -------------------- 3️⃣ Train-Test Split --------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['query'], df['label'], test_size=0.2, random_state=42
)

# -------------------- 4️⃣ Tokenization --------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the text data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, return_tensors='pt')

# -------------------- 5️⃣ Convert Labels to Torch Tensors --------------------
# -------------------- 5️⃣ Convert Labels to Torch Tensors --------------------
train_labels = torch.tensor(train_labels.values, dtype=torch.long)  # Convert to torch.long
val_labels = torch.tensor(val_labels.values, dtype=torch.long)  # Convert to torch.long


# -------------------- 6️⃣ Create PyTorch Datasets --------------------
train_dataset = TensorDataset(train_encodings['input_ids'], 
                              train_encodings['attention_mask'], 
                              train_labels)

val_dataset = TensorDataset(val_encodings['input_ids'], 
                            val_encodings['attention_mask'], 
                            val_labels)

# -------------------- 7️⃣ Create DataLoaders --------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# -------------------- 8️⃣ Load Model --------------------
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(le.classes_)  # Number of unique intents
)

# -------------------- 9️⃣ Setup Optimizer and Loss --------------------
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

# -------------------- 1️⃣0️⃣ Training Loop --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

# -------------------- 1️⃣1️⃣ Save Model and Label Encoder --------------------
model.save_pretrained("intent_classifier_pytorch")  # Save model
joblib.dump(le, "label_encoder.pkl")  # Save label encoder

print("Training complete and model saved!")

