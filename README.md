# Chefs Watch
Chefs watch is a food time prediction system. This was one of the challenges of the [Data Science Nexus Conference 2019](https://mailchi.mp/umanitoba.ca/data-science-conference) arranged by SkipTheDishes, Canada's leading and largest food delivery company. The data included upto 10 ordered food items along with the quantity. The goal is to predict food preparation time.

# Dataset info
The data included 20 features and 80,000 samples. Please check Data-Challenge-Skipthedishes-Repaired.pdf for details.

# Applied techniques
• Doc2Vec has been applied to vectorize food item names and feature engineering.<br/>
• RandomForest is used to predict food preparation time.<br/>
• Grid search is used for hyperparameter optimization.<br/>

# Result 
• The baseline performance is R2: 0.61, MAE: 5.45, RMSE: 6.98. The performance achieved with doc2Vec is R2:0.75, MAE: 4.41, RMSE: 5.56. The details can be found in performance.csv.

# Learning Curve
![](learning_curve.png)

# How to Run
Check Food_preparation_time_rf_doc2Vec.py
