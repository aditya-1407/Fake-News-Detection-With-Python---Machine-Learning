import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Data Loading and printing its various info
df = pd.read_csv('data.csv')
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

# Splitting the label and features
y = df.Label
# print(y.head())
df = df.drop('Label', axis=1)
# print(df.head())

# Handling missing values by filling them with an empty string
df['Body'].fillna('', inplace=True)
y.fillna('', inplace=True)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df['Body'], y, test_size=0.3, random_state=0)
# print(x_train.head())
# print(x_test.head())
# print(y_train.head())
# print(y_test.head())

# Feature Extraction
x_train = np.where(pd.isnull(x_train), '', x_train)
x_test = np.where(pd.isnull(x_test), '', x_test)

vectorizer = TfidfVectorizer()
train_vectorizer = vectorizer.fit_transform(x_train)
test_vectorizer = vectorizer.transform(x_test)

feature_names = vectorizer.get_feature_names_out()

count_df = pd.DataFrame(train_vectorizer.toarray(), columns=feature_names)
# print(count_df.head(10))

# Training the Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(train_vectorizer, y_train)

# Prediction & Accuracy Score
y_predict = model.predict(test_vectorizer)
# print(y_test.head())
# print(y_predict[:5])
acs = accuracy_score(y_test, y_predict)
print("accuracy : %0.3f"%acs)