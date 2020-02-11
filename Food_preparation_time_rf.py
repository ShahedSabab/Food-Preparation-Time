import csv 
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import regularizers
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.model_selection import learning_curve
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def performance(model, y_test, y_pred):
    model_name = str(model)
    model_name = model_name.split('(')
    model_name = model_name[0]
    df_temp=pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    r2 = metrics.r2_score(y_test, y_pred)
    d =[ model_name, mae , mse, rmse, r2]
    df_temp = df_temp.append(pd.Series(d,index=['Model', 'MAE', 'MSE', 'RMSE', 'R2']),ignore_index=True)
    return df_temp


def learning_curves(estimator, data, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
            estimator, data[features], data[target], train_sizes = train_sizes,
            cv = cv, scoring = 'neg_mean_absolute_error')
    train_scores_mean = -train_scores.mean(axis = 1)
    validation_scores_mean = -validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel('MAE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(estimator).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,8)

#two input files 
file_name = "train_orders_data_with_targets.csv"
food_name = "menu_items_details_complete.csv"

#testing perventage to split the data
test_portion = 0.20

df = pd.read_csv(file_name)
df_food = pd.read_csv(food_name)
df = df.drop('order_id', 1)
df = df.drop('datetime', 1)
categorical_columns = [f for f in df.columns if df[f].dtype == 'object']
df = df.fillna(0)


#encoding food id with numbers
labelEncoder = LabelEncoder()
df_food.insert(2,"encoded",labelEncoder.fit_transform(df_food['menu_item_id']))

food_item = df_food['title'].values



for c in categorical_columns:
    df[c] = df[c].map(df_food.set_index('menu_item_id')['encoded'])
df = df.fillna(0)    


# Convert the dataset into train and test datasets
train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,0:-1].values,df.iloc[:,-1].values, test_size=test_portion, random_state=4)


filename = 'rf_model.sav'
# load the model from disk
model = pickle.load(open(filename, 'rb'))

fileName="rf"
df_final = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2'])
#random forest regressor
#p = {'n_estimators':200, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth':10, 'bootstrap': 'True', 'oob_score': 'True', 'random_state': 100, 'n_jobs':-1, 'verbose':2}
#model = RandomForestRegressor(**p)
model.fit(train_x,train_y)
y_pred = model.predict(test_x)
df_final = df_final.append(performance(model, test_y, y_pred))
#
train_sizes = [1000, 2000, 5000, 10000, 30000, 50000, 65000, 72000]
learning_curves(model, df, df.iloc[:,0:-1].columns, 'food_prep_time_minutes', train_sizes, 10)
## save the model to disk
##pickle.dump(model, open(filename, 'wb'))
#
#