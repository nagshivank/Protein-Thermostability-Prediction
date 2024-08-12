import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch import nn, optim

# Function to embed sequences using Prot-BERT
def embed_sequences(sequences, tokenizer, model, max_length=512):
    inputs = tokenizer.batch_encode_plus(
        sequences, 
        add_special_tokens=True, 
        padding='max_length', 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Function to load data from CSV
def data_loader(csvpath):
    data = pd.read_csv(csvpath)
    data.sequence = data.sequence.apply(lambda s: re.sub(r"[^A-Z]", "", s.upper()))
    return data

# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Main function
def main():
    # Load the pretrained Prot-BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert')
    model = BertModel.from_pretrained('Rostlab/prot_bert')

    # Load data
    train_data = data_loader("train.csv")
    test_data = data_loader("test.csv")

    # Embedding sequences
    X_train = embed_sequences(train_data['sequence'].tolist(), tokenizer, model)
    X_test = embed_sequences(test_data['sequence'].tolist(), tokenizer, model)

    # Extracting target values
    y_train = train_data['target'].values

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the MLP model
    mlp_model = MLP(input_dim=X_train.shape[1])

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.05)

    # Training the model
    for epoch in range(100):
        mlp_model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = mlp_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        mlp_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = mlp_model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader)}, Validation Loss = {val_loss/len(val_loader)}")

    # Predicting on test data
    mlp_model.eval()
    with torch.no_grad():
        test_pred = mlp_model(X_test_tensor).numpy()

    # Saving the predictions
    predictions = pd.DataFrame({'id': test_data['id'], 'target': test_pred.flatten()})
    predictions.to_csv('mlp_prediction.csv', index=False)

if __name__ == "__main__":
    main()