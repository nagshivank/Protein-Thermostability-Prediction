# Protein Thermostability Prediction

## Overview

This repository contains the implementation of three machine learning models developed to predict the thermostability of proteins based on their amino acid sequences. The problem is part of a Kaggle competition that focuses on predicting protein thermostability, a key trait that enhances various application-specific functions in computational biology. 

### Problem Description

**Background:** Thermostability is a critical feature in proteins that influences their effectiveness in various applications, including enzymatic processes and directed evolution. Accurately predicting protein thermostability poses a significant challenge due to the complex biological factors involved, such as the impact of minor amino acid substitutions on stability. This competition invites participants to develop models capable of predicting protein thermostability using a curated dataset of protein sequences.

**Motivation:** Understanding and predicting thermostability is crucial for protein engineering, especially in scenarios where proteins need to operate under high-temperature conditions or are used as starting points for directed evolution. The ability to predict thermostability could lead to advancements in protein design, with implications for industrial and medical applications.

**Dataset:** The dataset for this challenge is curated from a mass spectrometry-based assay measuring protein melting curves. It includes a diverse set of protein sequences that exhibit both global and local variations, providing a comprehensive basis for understanding how sequence variations influence thermostability.

**Objective:** The goal is to develop machine learning models that can predict the thermostability of proteins. The models are evaluated based on the Spearman correlation between the predicted and actual thermostability values, with the final assessment made using a holdout test set.

## Repository Structure

This repository contains the following scripts:

1. `final_model_svr.py`
2. `mlp_prot_bert.py`
3. `ensemble_xgboost_rf.py`
4. `README.md`

### 1. Final Model: `final_model_svr.py`

#### Description
This script implements a Support Vector Regressor (SVR) with an RBF kernel to predict protein thermostability based on amino acid composition features. It was chosen as the final model due to its ability to handle high-dimensional data and model complex, non-linear relationships.

#### Features
- **Amino Acid Composition:** Calculates the proportion of each amino acid in the sequence.
- **Normalization:** Uses `StandardScaler` to normalize the amino acid composition features.
- **Cross-Validation:** Uses 5-fold cross-validation to evaluate model performance.
- **Spearman Correlation:** Custom function to calculate Spearman correlation as the performance metric.

#### How to Run
```bash
python final_model_svr.py
