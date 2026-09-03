# Predicting Employee Attrition Using Supervised and Unsupervised Machine Learning

## Project Overview
Employee attrition represents a significant financial and operational challenge for organizations. This repository contains a rigorous machine learning pipeline designed to predict employee attrition using historical Human Resources data. Developed for the CSE422 course at BRAC University, this project focuses on navigating severe class imbalance to produce mathematically stable and practical classification models.

## Dataset
The project utilizes an HR analytics dataset consisting of 1,677 employee records. The target variable is `Attrition` (0 = Stayed, 1 = Left). The data exhibits a severe 88:12 class imbalance, necessitating advanced resampling techniques to overcome the Majority Class Baseline (Zero-R).

## Methodology & Pipeline
1. **Exploratory Data Analysis (EDA) & Pre-processing:** 
   * Handled categorical variables using `LabelEncoder` and `OneHotEncoder`.
   * Applied `StandardScaler` for distance and gradient-based algorithm optimization.
2. **Multicollinearity Elimination:** 
   * Utilized **Variance Inflation Factor (VIF)** and **Spearman Rank Correlation** to reduce the feature space from 43 encoded features to 22 highly predictive variables.
3. **Imbalance Resolution:** 
   * Applied **SMOTETomek** strictly on the training data to synthesize minority class samples while removing noisy borderline points.
4. **Validation Strategy:** 
   * Implemented **Stratified 5-Fold Cross-Validation** to ensure the 88:12 class ratio was preserved across all folds, guaranteeing unbiased and robust evaluation metrics.

## Models Evaluated
* **Logistic Regression:** Achieved the highest aggregate AUC and F1-Score, serving as the most mathematically stable and interpretable model.
* **Feed-Forward Neural Network:** Constructed using Keras with Dense and Dropout layers. Demonstrated strong accuracy and class separation capabilities.
* **K-Nearest Neighbors (KNN):** Distance-based algorithm utilized to benchmark similarity-based attrition logic.
* **K-Means Clustering (Unsupervised):** Deployed as a baseline to demonstrate the necessity of historical labels, revealing that natural groupings (e.g., seniority) do not inherently map to flight risk.

## Cross-Validated Results (5-Fold Averages)

| Model | Accuracy | Precision | Recall | F1 Score | AUC Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.8437 | 0.4156 | 0.5350 | 0.4442 | 0.7945 |
| **Neural Network** | 0.8437 | 0.4301 | 0.4400 | 0.4036 | 0.7479 |
| **KNN** | 0.7448 | 0.2582 | 0.5250 | 0.3379 | 0.7064 |
| **K-Means (Unsupervised)**| 0.4061 | 0.1440 | 0.8050 | 0.2443 | 0.5785 |

## Repository Structure
* `Employee Attrition Prediction Model.ipynb`: The core Jupyter Notebook containing the data pre-processing, modeling, cross-validation, and visualization pipeline.
* `Dataset.csv`: The raw HR analytics dataset.
* `Report.pdf`: The comprehensive IEEE-formatted research report detailing the study's findings.
* `images/`: Directory containing output visualizations including Correlation Heatmaps, ROC Curves, and Confusion Matrices.

## How to Run
1. Download `Employee Attrition Prediction Model.ipynb` and `Dataset.csv`.
2. Open the notebook in Google Colab or a local Jupyter environment.
3. Ensure the dataset is in the same working directory.
4. Execute the notebook sequentially to reproduce the pre-processing, 5-fold training, and evaluation outputs.
