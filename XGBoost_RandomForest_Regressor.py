import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr
import re

# Function to calculate k-mer features
def kmer_features(peptides, k=3):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    features = vectorizer.fit_transform(peptides)
    return features.toarray()

# Function to calculate hydrophobicity
def calculate_hydrophobicity(peptides):
    hydrophobicity_scale = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
        'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
        'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
        'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
    }
    hydrophobicity = []
    for seq in peptides:
        score = np.mean([hydrophobicity_scale.get(aa, 0) for aa in seq])
        hydrophobicity.append(score)
    return np.array(hydrophobicity).reshape(-1, 1)

# Function to load data from CSV
def data_loader(csvpath):
    data = pd.read_csv(csvpath)
    data.sequence = data.sequence.apply(lambda s: re.sub(r"[^A-Z]", "", s.upper()))
    return data

# Function to calculate Spearman correlation coefficient
def spearman(y_true, y_pred):
    coeff, _ = spearmanr(y_true, y_pred)
    return coeff

# Main function
def main():
    # Load data
    train_data = data_loader("train.csv")
    test_data = data_loader("test.csv")

    # Calculate k-mer features and hydrophobicity
    kmer_train = kmer_features(train_data['sequence'].tolist())
    kmer_test = kmer_features(test_data['sequence'].tolist())
    hydrophobicity_train = calculate_hydrophobicity(train_data['sequence'].tolist())
    hydrophobicity_test = calculate_hydrophobicity(test_data['sequence'].tolist())

    # Scaling hydrophobicity
    scaler = MinMaxScaler()
    hydrophobicity_train = scaler.fit_transform(hydrophobicity_train)
    hydrophobicity_test = scaler.transform(hydrophobicity_test)

    # Concatenating features
    X_train = np.hstack([kmer_train, hydrophobicity_train])
    X_test = np.hstack([kmer_test, hydrophobicity_test])
    y_train = train_data['target'].values

    # Train-test split for validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize the models
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=42)

    # Train the models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Predict on validation data and calculate the average of both predictions
    rf_val_pred = rf_model.predict(X_val)
    xgb_val_pred = xgb_model.predict(X_val)
    ensemble_val_pred = (rf_val_pred + xgb_val_pred) / 2

    # Spearman correlation
    val_score = spearman(y_val, ensemble_val_pred)
    print(f"Validation Spearman Correlation: {val_score}")

    # Predict on test data and average predictions
    rf_test_pred = rf_model.predict(X_test)
    xgb_test_pred = xgb_model.predict(X_test)
    ensemble_test_pred = (rf_test_pred + xgb_test_pred) / 2

    # Save predictions
    predictions = pd.DataFrame({'id': test_data['id'], 'target': ensemble_test_pred})
    predictions.to_csv('ensemble_prediction.csv', index=False)

if __name__ == "__main__":
    main()