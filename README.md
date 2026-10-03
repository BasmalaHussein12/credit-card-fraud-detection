# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using exploratory data analysis, classical classification models, and a neural network.

## Project Overview

Credit card fraud detection is a binary classification problem where the goal is to distinguish between legitimate and fraudulent transactions.

This project focuses on understanding the dataset through exploratory data analysis (EDA), preparing the data appropriately for machine learning, and comparing traditional classification models with a neural network.

## Dataset

The dataset contains credit card transactions with the following main features:

* `Time` — elapsed time since the first transaction in the dataset.
* `V1`–`V28` — anonymized PCA-transformed features.
* `Amount` — transaction amount.
* `Class` — target variable:

  * `0` → Legitimate transaction
  * `1` → Fraudulent transaction

The dataset is highly imbalanced, with fraudulent transactions representing a very small proportion of the total transactions.

## Exploratory Data Analysis

The EDA focused on understanding the structure of the data, class distribution, feature distributions, temporal patterns, and relationships between features and the target.

### Key Findings

* The dataset is highly imbalanced, with fraud representing a small minority of transactions.
* No missing values were found.
* Duplicate records were checked.
* `Amount` has a skewed distribution with extreme values.
* Fraudulent and legitimate transactions show noticeable differences in the distributions of several `V` features.
* Fraudulent transactions occur across multiple hours rather than being concentrated in a single time period.
* Fraud rates vary across hours, with the highest rate observed at Hour 2 (1.71%), followed by Hour 4 (1.04%).
* Correlation analysis showed that `V17`, `V14`, `V12`, and `V10` have the strongest linear associations with the target.
* Several features have weak or near-zero linear correlations with `Class`; however, weak correlation does not necessarily mean that a feature is uninformative, since nonlinear relationships and feature interactions may still contain useful information.

### PCA-Transformed Features

The `V1`–`V28` features are anonymized PCA-transformed features provided in the dataset. PCA reduces the original feature space into new components that are largely uncorrelated with one another.

Because the original feature meanings are not provided, the `V` features are treated as numerical predictive features rather than being assigned specific real-world interpretations.

## Feature Selection

The initial modeling feature set consists of:

* `V1`–`V28`
* `Amount`
* `Hour`

The original `Time` feature is excluded initially because `Hour` was derived from the transaction time and provides a more direct representation of time-of-day patterns.

Features are not removed solely based on their individual correlation with the target, since useful predictive information may exist through nonlinear relationships and interactions between multiple features.

## Machine Learning Approach

The project will compare different classification approaches.

### Baseline Models

Classical machine learning models will be used as baselines, such as:

* Logistic Regression
* Random Forest

---------------------------------------------------------------------------------------------------------------

## Feature Importance & Model Interpretation

One of the most interesting parts of this project was exploring **feature importance** for the first time.

After training my first machine learning model, the **Random Forest**, I examined which features contributed the most to its predictions:

```python
feature_importance = pd.Series(
    Random_Forest_Model.feature_importances_,
    index=x_train.columns
).sort_values(ascending=False)

feature_importance
```

The most important features were:

| Feature | Importance |
| ------- | ---------: |
| V17     |     19.91% |
| V14     |     13.29% |
| V12     |     10.05% |
| V11     |      7.31% |
| V10     |      7.13% |

### EDA Findings Confirmed by the First ML Model

One of the most exciting outcomes was seeing that my **EDA analysis was supported by my first machine learning model**.

During EDA, I investigated the relationship between the features and the target `Class`. The features `V17`, `V14`, `V12`, `V11`, and `V10` showed some of the strongest relationships with the target.

After training the Random Forest, these same features appeared among the **most important features used by the model**.

I also visualized their distributions across legitimate and fraudulent transactions and found noticeable differences between the two classes.

This gave me a strong progression:

**EDA → Identified potentially important features → Visualized their distributions → Trained the first ML model → Feature importance supported the EDA findings**

This was especially meaningful because the model was trained independently of my earlier conclusions. It gave me evidence that the patterns I noticed during EDA were not simply observations in isolation — they contained useful predictive information for the classification task.

> **Note:** Correlation and Random Forest feature importance measure different things. Correlation measures the strength of a linear relationship with the target, while feature importance reflects how useful a feature was to the Random Forest when making predictions. Therefore, their values should not be compared directly.

### A Personal Learning Moment (I am over the moon due to this project)

This was my **first time using feature importance**, and honestly, it blew my mind.

Seeing the Random Forest independently identify `V17`, `V14`, `V12`, `V11`, and `V10` as highly important after I had already noticed similar patterns during EDA was one of the most exciting moments of this project.

It made the connection between **data exploration and machine learning** feel much more real to me.

For the first time, I wasn't just:

> *"I trained a model and got 99.96% accuracy."*

I was actually asking:

> *"What did my model learn, and does it agree with what I discovered during EDA?"*

And seeing that my **EDA analysis was supported by my first ML model** was a genuinely rewarding moment in my learning journey.

> **Important:** Feature importance does not mean that these features causally cause fraud. It indicates that the Random Forest relied more heavily on these features when making its predictions.

---------------------------------------------------------------------------------------------------------------

### Neural Network

A neural network will then be developed as the main deep learning model for the binary classification task.

The comparison will help evaluate how the neural network performs relative to traditional machine learning approaches.

## Preprocessing

The preprocessing pipeline will include:

1. Separating features (`X`) from the target (`y`).
2. Splitting the data into training, validation, and test sets.
3. Scaling numerical features where appropriate.
4. Addressing the severe class imbalance using an appropriate strategy.
5. Ensuring that preprocessing steps are fitted only on the training data to prevent data leakage.

## Model Evaluation

Because this dataset is highly imbalanced, accuracy alone is not sufficient for evaluating fraud detection performance.

The models will be evaluated using metrics such as:

* Precision
* Recall
* F1-score
* Confusion Matrix
* Precision-Recall AUC (PR-AUC)

Particular attention will be given to **recall for the fraud class**, since missing fraudulent transactions can be costly in a fraud detection system.

## Project Structure

```text
credit-card-fraud-detection/
│
├── images/
│   ├── Distribution of Transaction Amount.png
│   ├── Transaction Count By Hour (Legitimate vs Fraud).png
│   ├── transaction_distribution_by_hour.png
│   ├── fraud_rate_by_hour.png
│   └── correlation_heatmap.png
│
├── notebook/
│   └── project.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

> The original dataset is not included in the repository.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

## Current Status

* [x] Data exploration
* [x] Class imbalance analysis
* [x] Missing-value and duplicate checks
* [x] Transaction amount analysis
* [x] Time-based analysis
* [x] Feature-target correlation analysis
* [x] Initial feature selection
* [ ] Data preprocessing
* [ ] Baseline classification models
* [ ] Neural network
* [ ] Model evaluation
* [ ] Model comparison

## Goal

The goal of this project is to build a reliable fraud detection classification pipeline while developing a deeper understanding of:

* Exploratory data analysis
* Feature preprocessing
* Imbalanced classification
* Classical machine learning
* Neural networks
* Model evaluation for fraud detection
* Interpreting model behavior


`Basmala Hussein`
**An aspiring ML Engineer**