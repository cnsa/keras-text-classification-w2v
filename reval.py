# coding=utf-8
from numpy import random, arange, float32, ndenumerate

import h5py, os

import seaborn as sns

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import load_model

from data_helper import print_predictions, load
from web_data import data_folder_path
from webhose_data import load_data_and_labels

# Set Numpy random seed
random.seed(1000)

# Newsline folder and format
data_folder = data_folder_path('cnsa')

# Word2Vec number of features
num_features = 500
# Limit each newsline to a fixed number of words
document_max_num_words = 100

model_name = "cnsa.model.h5"
X_train, X_test, Y_train, Y_test, idx_train, idx_test, document_X_title = load(data_folder)

model = load_model(data_folder + model_name)

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

predicted = model.predict(X_test)

for (x, y), value in ndenumerate(predicted[:100]):
    if value < 0.3:
        predicted[x, y] = 0.0
    else:
        predicted[x, y] = value

    if value >= 0.3:
        predicted[x, y] = 1.0
    else:
        predicted[x, y] = value

scatter_matrix(pd.DataFrame(predicted[:10]), alpha=0.2, figsize=(6, 6))
scatter_matrix(pd.DataFrame(Y_test[:10]), alpha=0.2, figsize=(6, 6))

p1 = pd.DataFrame(predicted[:10]).plot(title=u"Предсказанные результаты")
p1.set_xlabel('Text')
p1.set_ylabel('Probability')
p2 = pd.DataFrame(predicted[:10]).plot(title=u"Предсказанные результаты", kind="box")
p2.set_xlabel('Factor')
p2.set_ylabel('Probability')

p3 = pd.DataFrame(Y_test[:10]).plot(title=u"Актуальные результаты")
p3.set_xlabel('Text')
p3.set_ylabel('Probability')
p4 = pd.DataFrame(Y_test[:10]).plot(title=u"Актуальные результаты", kind="box")
p4.set_xlabel('Factor')
p4.set_ylabel('Probability')

df = pd.DataFrame(predicted[:100])
df['Factor'] = pd.Series(range(predicted.shape[1]))
sns.set(style="ticks")
sns.pairplot(df, hue="Factor")

df = pd.DataFrame(Y_test[:100])
df['Factor'] = pd.Series(range(Y_test.shape[1]))
sns.set(style="ticks")
sns.pairplot(df, hue="Factor")

# Show the plot

# print_predictions(predicted, document_X_title, selected_categories, idx_test, y=y_train_text, show_words=100, encode=False)

print('Loss: %1.4f' % score)
print('Accuracy: %1.4f' % acc)

plt.show()
