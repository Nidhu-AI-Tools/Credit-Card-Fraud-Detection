# Credit Card Fraud Detection (End-to-End ML Project)

This project is my attempt at building a full machine learning pipeline on a real-world problem: detecting fraudulent credit card transactions.

Instead of treating this as a toy problem, I’m approaching it as if it were something closer to production. That means dealing with messy realities like class imbalance, evaluation tradeoffs, and model behavior beyond just accuracy.

---

## Problem

Fraud detection is fundamentally an imbalanced classification problem. In this dataset, fraudulent transactions make up a tiny fraction of all observations. A naive model can achieve very high accuracy by simply predicting "no fraud" for everything, which makes accuracy a misleading metric.

The goal is not just to build a model, but to understand:
- How models behave under extreme imbalance
- What tradeoffs exist between precision and recall
- How to evaluate models in a way that reflects real-world costs

---

## Dataset

The dataset used is the European cardholders fraud detection dataset, commonly available on Kaggle.

Key characteristics:
- ~284,000 transactions
- Highly imbalanced (~0.17% fraud)
- Features are anonymized (PCA transformed: V1–V28)
- Additional features: `Time`, `Amount`
- Target variable: `Class` (0 = legitimate, 1 = fraud)

Because of anonymization, this project focuses more on modeling and evaluation rather than domain-specific feature interpretation.

---

## Project Structure

fraud-ml/
│── data/               # raw and processed data

│── notebooks/          # exploratory work and experiments
│── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│── models/             # saved models
│── outputs/            # metrics, plots, logs

---

## Approach

### 1. Data Exploration
- Understand class distribution
- Check for missing values and basic statistics
- Identify any obvious issues with the dataset

### 2. Baseline Model
- Start with a simple model (logistic regression)
- Establish a baseline using appropriate metrics

### 3. Evaluation Strategy
Given the imbalance, the focus is on:
- Precision
- Recall
- F1 Score
- ROC-AUC

Accuracy is not used as a primary metric.

### 4. Handling Imbalance
Different strategies explored:
- Class weighting
- Oversampling (e.g., SMOTE)
- Undersampling
- Threshold tuning

### 5. Model Iteration
- Tree-based models (Random Forest, Gradient Boosting)
- Compare performance across models
- Focus on tradeoffs rather than just maximizing a single metric

### 6. Toward Production
- Save trained models
- Build a simple prediction interface
- Simulate real-world usage (incoming transaction scoring)

---

## Key Questions Driving This Project

- What is the cost of false negatives vs false positives in this context?
- How far can recall be pushed before precision becomes unusable?
- How stable are results across different splits?
- How much improvement comes from modeling vs data handling?

---

## Current Status

Work in progress. The goal is not just to get a good score, but to understand the system end-to-end and where things break.

---

## Notes

This project is intentionally iterative. The focus is on learning through building, testing, and refining rather than following a fixed template.
