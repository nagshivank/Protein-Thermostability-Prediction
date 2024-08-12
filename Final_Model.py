import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import re
from sklearn.metrics import make_scorer

# Function to calculate the amino acid composition for each peptide
def aa_composition(peptides):

    # Standard Amino Acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    features = np.zeros((len(peptides), len(amino_acids)))
    # Calculating the frequency of each amino acid in each peptide
    for i, seq in enumerate(peptides):
        for j, aa in enumerate(amino_acids):
            features[i, j] = seq.count(aa) / len(seq)
    return features

# Function to load data from the CSV files
def data_loader(csvpath):
    data = pd.read_csv(csvpath)
    # Cleaning the sequences to ensure it only contains uppercase amino acid symbols
    data.sequence = data.sequence.apply(lambda s: re.sub(r"[^A-Z]", "", s.upper()))
    return data

# Function to calculate Spearman correlation coefficient
def spearman(y_true, y_pred):
    coeff, _ = spearmanr(y_true, y_pred)
    return coeff

# The main function
def main():
    # Loading the training and test data
    train_data = data_loader("train.csv")
    test_data = data_loader("test.csv")

    # Calculating amino acid composition features training and test datasets
    X_train = aa_composition(train_data['sequence'].tolist())
    X_test = aa_composition(test_data['sequence'].tolist())

    # Normalizing the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Extracting target values from training dataset
    y_train = train_data['target'].values

    # Designing the k-fold cross-validation setup
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

    # Initializing the SVR model with RBF kernel
    model = SVR(kernel = 'rbf', C = 1.0, epsilon = 0.1)

    # Performing cross-validation and calculating correlation for each fold
    scores = cross_val_score(model, X_train, y_train, cv = kf, scoring = make_scorer(spearman))

    # Printing the average Spearman correlation across all folds to console
    print(f"Average Spearman Correlation from Cross-Validation: {np.mean(scores)}")

    # Training the model on the complete training set to predict test targets
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)

    # Saving the predictions to a CSV file
    predictions = pd.DataFrame({'id': test_data['id'], 'target': test_pred})
    predictions.to_csv('prediction.csv', index=False)

if __name__ == "__main__":
    main()