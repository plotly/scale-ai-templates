#!/usr/bin/env python
# coding: utf-8

# To train this model, please download [the dataset from Kaggle](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018?select=drugsComTrain_raw.csv), and extract `drugsComTrain_raw.csv` in this folder.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVR, SVR, LinearSVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
import joblib
import plotly.express as px


# ## Load training data

# In[ ]:


train = pd.read_csv("drugsComTrain_raw.csv")


# ## Create and train vectorizer/svd

# In[ ]:


vectorizer = TfidfVectorizer(
    stop_words="english", max_df=0.95, min_df=5, max_features=5000
)
reducer = TruncatedSVD(n_components=1000, n_iter=5)

vecs = vectorizer.fit_transform(train.review)
vecs = reducer.fit_transform(vecs)


# ## Train Validation Split

# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(
    vecs, train.rating.values, test_size=0.1, random_state=2020
)


# ## Train model

# In[ ]:


model = LogisticRegression()

model.fit(X_train, y_train)


# ## Evaluate trained model

# In[ ]:


train_r2 = model.score(X_train, y_train)
val_r2 = model.score(X_val, y_val)

train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

train_f1 = f1_score(y_train, train_pred, average="micro")
val_f1 = f1_score(y_val, val_pred, average="micro")

print("Train r2:", train_r2)
print("Validation r2:", val_r2)

print("Train f1:", train_f1)
print("Validation f1:", val_f1)


# ## Save everything

# In[ ]:


joblib.dump(model, "review_model/model.joblib")
joblib.dump(vectorizer, "review_model/vectorizer.joblib")
joblib.dump(reducer, "review_model/reducer.joblib")
