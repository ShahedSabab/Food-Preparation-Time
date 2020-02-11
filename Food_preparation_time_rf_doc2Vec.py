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
embeddedVectorSize = 10
windowSize = 2
epoch_doc2vec = 200


#preprocess the features and drop unimportant features
df = pd.read_csv(file_name)
df_food = pd.read_csv(food_name)
df = df.drop('order_id', 1)
df = df.drop('datetime', 1)
categorical_columns = [f for f in df.columns if df[f].dtype == 'object']
df = df.fillna(0)

#encoding food id with numbers using label encoder
labelEncoder = LabelEncoder()
df_food.insert(2,"encoded",labelEncoder.fit_transform(df_food['menu_item_id']))

food_item = df_food['title'].values


#tokenize data
tokenized_item = []

for t in food_item:
    tokenized_item.append(word_tokenize(t.lower())) 

#Convert tokenized data into gensim formatted tagged data
tagged_data = [TaggedDocument(d,[i]) for i, d in enumerate(tokenized_item)] 

embedded_model = Doc2Vec(tagged_data, vector_size=embeddedVectorSize, window=windowSize, min_count = 2, worker=4, epoch= epoch_doc2vec)

#save the trained model 
embedded_model.save("doc2vec.model")

embedded_model = Doc2Vec.load("doc2vec.model")


#appending the feature vector to the dataframe
encodedw2v=[]
    
for i in range(0,df_food.shape[0]):
    encodedw2v.append(list(embedded_model.docvecs[i]))

df_food.insert(2,"encodedw2v",encodedw2v)
   
loc = []
for c in categorical_columns:
    loc.append(df.columns.get_loc(c))
    df[c] = df[c].map(df_food.set_index('menu_item_id')["encodedw2v"])
    df[c].fillna(0, inplace=True)


col=[[],[],[],[],[],[],[],[],[],[]]
for i in range(len(categorical_columns)):
    for j in range(embeddedVectorSize):
        col[i].append('item_'+repr(i+1)+repr(j+1))

df = df.fillna(0)
fill_val = np.zeros(embeddedVectorSize)

for c in categorical_columns:
    for i, val in enumerate(df[c]):
        if val==0:
            df.at[i, c] = fill_val

for i in range(len(col)):
    df[col[i]] = pd.DataFrame(df['item_'+repr(i+1)].values.tolist())

col_df = []
for i in range(len(col)):
    col_df.extend(col[i])
    col_df.append("quantity_"+repr(i+1))    
col_df.append('food_prep_time_minutes')

df = df[col_df]

train_features = df.iloc[:,0:-1].values
train_label = df.iloc[:,-1].values


# Convert the dataset into train and test datasets
train_x, test_x, train_y, test_y = train_test_split(train_features, train_label, test_size=test_portion, random_state=4)


filename = 'rf_model2.sav'
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
###
train_sizes = [1000, 2000, 5000, 10000, 30000, 50000, 65000, 72000]
learning_curves(model, df, df.iloc[:,0:-1].columns, 'food_prep_time_minutes', train_sizes, 10)
#save the model to disk
#pickle.dump(model, open(filename, 'wb'))
#
#

