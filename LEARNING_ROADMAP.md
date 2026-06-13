# Learning Roadmap

This roadmap is tied to this credit card fraud detection project.

The goal is not to learn every ML topic at once. The goal is to learn the right things in the right order while building the project.

## Phase 1: Foundations You Need Right Now

### 1. Train / Validation / Test Split
- What to learn:
  How and why data is split before model training.
- Why it matters here:
  Without proper splits, model evaluation is unreliable.
- When to learn it:
  Before training any model.
- What to practice in this project:
  Use stratified `75 / 15 / 10` splitting and verify the fraud ratio in each split.

### 2. Stratification
- What to learn:
  Why imbalanced datasets need class proportions preserved across splits.
- Why it matters here:
  Fraud is very rare, so bad splits can distort results.
- When to learn it:
  Along with data splitting.
- What to practice in this project:
  Compare overall fraud proportion with train, validation, and test fraud proportions.

### 3. Features, Target, and Labels
- What to learn:
  The difference between input features and the target variable.
- Why it matters here:
  The model should learn from transaction features and predict `Class`.
- When to learn it:
  Before fitting any model.
- What to practice in this project:
  Keep `X` and `y` clearly separated in the training notebook.

### 4. Data Leakage
- What to learn:
  What leakage is and how it quietly invalidates model evaluation.
- Why it matters here:
  Fraud detection results can look better than they really are if information leaks from validation or test data.
- When to learn it:
  Immediately.
- What to practice in this project:
  Do not tune models, thresholds, or sampling methods using the test set.

## Phase 2: Learn How to Evaluate an Imbalanced Classification Problem

### 5. Confusion Matrix
- What to learn:
  The meaning of true positives, false positives, true negatives, and false negatives.
- Why it matters here:
  Fraud detection is really about understanding what kinds of mistakes the model makes.
- When to learn it:
  Right after the dummy baseline.
- What to practice in this project:
  Read the confusion matrix for the dummy baseline, logistic regression, and decision tree.

### 6. Precision, Recall, and F1 Score
- What to learn:
  What each metric means and how they trade off against each other.
- Why it matters here:
  Fraud detection is not about being generally correct. It is about balancing missed frauds and false alarms.
- When to learn it:
  During model evaluation.
- What to practice in this project:
  Explain in words what each model is optimizing based on its precision and recall pattern.

### 7. ROC-AUC vs PR-AUC
- What to learn:
  Why ROC-AUC can look strong even when a model is weak for rare-event detection, and why PR-AUC is more important here.
- Why it matters here:
  Your project is heavily imbalanced, so PR-AUC should guide model comparison.
- When to learn it:
  Right after your first real baseline results.
- What to practice in this project:
  Compare logistic regression and the decision tree mainly through PR-AUC, not just ROC-AUC.

### 8. Why Accuracy Is Not a Good Main Metric Here
- What to learn:
  Why high accuracy can still mean a useless fraud detector.
- Why it matters here:
  The dummy baseline already shows this.
- When to learn it:
  During baseline evaluation.
- What to practice in this project:
  Explain why predicting all non-fraud is not acceptable even though it would appear accurate.

## Phase 3: Learn the First Real Modeling Workflow

### 9. Dummy Baseline
- What to learn:
  Why every ML workflow needs a simple reference point.
- Why it matters here:
  It gives you a minimum standard that a real model must beat.
- When to learn it:
  Before training logistic regression.
- What to practice in this project:
  Interpret why the dummy baseline has `recall = 0`, `F1 = 0`, and `ROC-AUC = 0.5`.

### 10. Logistic Regression
- What to learn:
  How logistic regression works at a high level as a linear probabilistic classifier.
- Why it matters here:
  It is your first real baseline and already performs strongly on this project.
- When to learn it:
  Right now.
- What to practice in this project:
  Understand why logistic regression produces both class predictions and fraud probabilities.

### 11. Standardization and Scaling
- What to learn:
  Why some models care about feature scale and how scaling affects optimization.
- Why it matters here:
  `Time` and `Amount` are on very different scales from the PCA features.
- When to learn it:
  While working with logistic regression.
- What to practice in this project:
  Understand why `StandardScaler()` was used in the pipeline.

### 12. Pipelines
- What to learn:
  How preprocessing and model training are chained together safely using scikit-learn pipelines.
- Why it matters here:
  Pipelines reduce mistakes and make the workflow reproducible.
- When to learn it:
  Along with logistic regression.
- What to practice in this project:
  Understand each step in the logistic regression pipeline and why it belongs there.

## Phase 4: Learn Model Behavior and Tradeoffs

### 13. Decision Trees
- What to learn:
  How trees split data and why they often behave differently from linear models.
- Why it matters here:
  Your decision tree achieved much higher recall but extremely poor precision.
- When to learn it:
  After logistic regression.
- What to practice in this project:
  Explain why the tree behaves like an aggressive fraud detector.

### 14. Overfitting vs Underfitting
- What to learn:
  How models can be too simple or too flexible.
- Why it matters here:
  Trees can overreact to minority-class patterns if not controlled.
- When to learn it:
  While comparing logistic regression and the tree model.
- What to practice in this project:
  Relate `max_depth=5` to controlling tree complexity.

### 15. Class Weights
- What to learn:
  How models can be told to care more about the minority class.
- Why it matters here:
  Fraud cases are rare, so weighting can improve recall without changing the dataset itself.
- When to learn it:
  Before trying resampling methods.
- What to practice in this project:
  Compare weighted and unweighted model behavior later.

## Phase 5: Learn the Next Important Improvement

### 16. Threshold Tuning
- What to learn:
  How probability scores are converted into class predictions and why the default threshold is not always correct.
- Why it matters here:
  This is likely your next most useful improvement after logistic regression.
- When to learn it:
  Next.
- What to practice in this project:
  Test different probability thresholds on the validation set and observe how precision and recall move.

### 17. Precision-Recall Tradeoff
- What to learn:
  Why improving recall usually reduces precision, and vice versa.
- Why it matters here:
  Fraud systems often need a deliberate balance rather than a default threshold.
- When to learn it:
  Together with threshold tuning.
- What to practice in this project:
  Decide whether your project should prioritize catching more fraud or reducing false alarms.

## Phase 6: Learn More Advanced Imbalance Handling

### 18. Oversampling and Undersampling
- What to learn:
  How resampling changes the class balance in the training data.
- Why it matters here:
  These methods can help models see more fraud examples, but they must be used carefully.
- When to learn it:
  After threshold tuning.
- What to practice in this project:
  Learn why resampling must happen only on training data, never before splitting.

### 19. SMOTE
- What to learn:
  How synthetic minority examples are created.
- Why it matters here:
  It is a common method in imbalanced classification, but it can also create misleading patterns if used carelessly.
- When to learn it:
  After understanding simpler methods first.
- What to practice in this project:
  Compare SMOTE-based training later against class weighting.

## Phase 7: Learn Stronger Models

### 20. Random Forest
- What to learn:
  How an ensemble of trees improves over a single decision tree.
- Why it matters here:
  It is a natural next step after your simple tree baseline.
- When to learn it:
  After threshold tuning and class-weight experiments.
- What to practice in this project:
  Compare it against logistic regression, not just against the decision tree.

### 21. Gradient Boosting / XGBoost / LightGBM
- What to learn:
  Why boosted trees are often very strong on tabular data.
- Why it matters here:
  These models may outperform both logistic regression and a single tree.
- When to learn it:
  After you are comfortable with baseline evaluation and tradeoffs.
- What to practice in this project:
  Use them only after you can explain what your baseline models are doing.

## Phase 8: Learn Good ML Habits

### 22. Fair Model Comparison
- What to learn:
  How to compare models using the same data split and same metrics.
- Why it matters here:
  Otherwise “better” results can just be artifacts of process differences.
- When to learn it:
  Always.
- What to practice in this project:
  Keep the validation set fixed while comparing models.

### 23. Reproducibility
- What to learn:
  Why `random_state` and consistent notebooks matter.
- Why it matters here:
  You want your experiments to be rerunnable and comparable.
- When to learn it:
  Immediately.
- What to practice in this project:
  Keep random seeds fixed in splits and models.

### 24. Final Test Set Discipline
- What to learn:
  Why the test set should be touched only at the end.
- Why it matters here:
  Using the test set during experimentation turns it into another validation set.
- When to learn it:
  Before you evaluate your final chosen model.
- What to practice in this project:
  Do not use test results to decide thresholds or pick models.

## Recommended Next Steps In This Project

1. Learn threshold tuning.
2. Learn how precision-recall tradeoffs change with threshold.
3. Try logistic regression with threshold adjustment.
4. Then try class weighting or a stronger tree-based model.
5. Only after that move to more advanced imbalance methods like SMOTE.

## What To Ignore For Now

- Implementing models from scratch
- Deep learning for this dataset
- Very advanced hyperparameter tuning
- Complex feature engineering
- Deployment details

These are not useless. They are just not the highest-value learning steps for this project right now.
