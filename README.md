# Retail-Customer-Classification-Modelling
 Classification models for predicting customer outcomes in an unbalanced classification setting, with outcomes dependent on customer demographics (age, post_code, gender) as well shopping frequencies and average spends by department etc.
 
### PURPOSE OF PROJECT: 
- The overarching purpose of these models (this project) is to determine why some loyalty customers have chosen to opt into email marketing and others have chosen to opt out. The data shows that those customers who opt in have higher average spends, thus it is important to determine whether customers who are already spending more choose to opt in or whether opting in results in a higher average spend. To determine this, I construct various data features which capture demographics of customers and utilise various classification models to determine outcomes based off the input demographics / data features. A desired takeaways is to understand what features underpin each group and inform decisions on how to influence customers to opt into email marketing.

#### THE DATA:
- Original data is 52K entries of sales data across 12+ departments and several hundred unique customers with unique individual characteristics: post code, age, gender, shopping frequency and average spend across different departments. The data ranges from 2019 to early 2021 (~ 2Y )

#### PROJECT FLOW:
After initial data exploration and data cleaning, I create a variety of features for the classification models:
- Customer Duration (Time between first and last transactions)
- One-Hot Encode: post code data by customer, customer gender, and department sales frequencies by customer
- Standardise these variables without mean (Ex. post-code) to preserve the sparse matrix nature of the data + (the age variable)

A severe class imbalance problem was present between customers who opted into email marketing (majority class) vs those who didn't (minority class). To remedy this, I upscaled the minority class with replacement to balance the two classes
- Also tried downscaling the majority class in a separate iteration and found the former method to be superior

I first run the following models before tuning and compare performance (accuracy) across the average of (5) cross validations on the training data:
- Naive Bayes (baseline measurement)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-nearest neighbors
- Support Vector Classifier
- XGBoost Classifier
- Soft & Hard Voting Classifiers

#### the final result is the tuned support vector classifier as the clear winner with 90% accuracy on the training data and ~96% on the test data
