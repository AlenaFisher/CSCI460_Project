Predicting Air Pressure System (APS) Failures in Scania Trucks

Course: CSCI 460 - Introduction to Applied Machine Learning
Author: Alena Fisher
Date: May 3, 2026

Overview
This project applies machine learnign techniques to predict failures in the
Air Pressure System (APS) of heavy Scania trucks. The APS is reponsible for
the braking and gear shifting systems. Undetected failures can lead to costly
repairs and serious safety risks. Using sensor data collected during normal truck
operation, six machine learning models were trained and evaluated on a binary
classification task: predicting whether a given instance represents and APS
failure or not.

The primary challenge of this dataset is the severe class imbalance, where only 1.67% of training instances are actual APS failures. This is why the F1-score was chosen as the evaluation metric than accuracy alone.

Dataset
APS Failure at Scania Trucks - IDA 2016 Industrial Challenge
- Target variable: class - pos (APS failure) or neg (not APS failure)
- Features: 171 anonymized numerical sensor measurements
- Class distribution: 98.33% negative, 1.67% positive

Project Structure
- Fisher_Alena_Report.pdf
    - Comprehensive technical report
- Fisher_Alena_Poster.pdf
    - Research poster
- Fisher_Alena_Video.txt
    - Link to video presentation
- Fisher_Alena_Project.ipynb
    - Google Colab notebook with all code

Methods
1. Preprocessing
    - Replace all "na" strings with NaN
    - Converted all columns to numeric types
    - Removed features missing more than 80% of data
    - Median imputation (fitted on training set only)
    - Log1p transofmraiton to reduce skewness
    - StandardScaler standardization

2. Models Trained
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest
    - XGBoost

All models were tuned using GridSearchCV with k-fold cross-validation, with F1-score as the main evaluation metric.

Results
	Model	                Train F1       Test Accuracy	Test Precision	Test Recall	  Test F1
5	XGBoost	                0.859459          0.99850	       0.991080	       1.000	  0.995520
4	Random Forest	        0.808511          1.000000	       1.000000	       1.000	  1.000000
3	SVM	                    0.786227          0.998017	       0.894360	       0.999	  0.943788
2	KNN	                    0.774194          1.000000	       1.000000	       1.000	  1.000000
1	Decision Tree	        0.711811          0.995333	       0.947761	       0.762	  0.844789
0	Logistic Regression	    0.556250          0.962933	       0.306206	       0.967	  0.465127


XGBoost had the best overall performance with a test F1-score of 0.8585 and the fewest false negatives (57) of all models, making it the most suitable models for this classification task.

Requirements
- python 3.8
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

Install dependencies:
pip install pandas numpy scikit-learn xgboost maplotlib seaborn

Usage
1. Download the dataset from https://archive.ics.uci.edu/dataset/414/ida2016challenge
2. Place aps_failure_training_set.csv and aps_failure_test_set.csv in the proejct directory.
3. Open and run Fisher_Alena_Report.ipynb in Google Colab.