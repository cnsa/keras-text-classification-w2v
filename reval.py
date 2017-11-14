# coding=utf-8
from numpy import random, arange, float32, ndenumerate, tile, repeat, c_

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


def probability_plot(data, title=None, x_title=None, y_title=None, kind="line"):
    p = pd.DataFrame(data).plot(title=title, kind=kind)
    p.set_xlabel(x_title)
    p.set_ylabel(y_title)


probability_plot(predicted[:10], u"Предсказанные результаты", 'Text', 'Probability')
probability_plot(predicted[:10], u"Предсказанные результаты", 'Factor', 'Probability', kind="box")

probability_plot(Y_test[:10], u"Актуальные результаты", 'Text', 'Probability')
probability_plot(Y_test[:10], u"Актуальные результаты", 'Factor', 'Probability', kind="box")

sns.set(style="ticks")


def facet_grid(data, limit=None, title="Figure"):
    if limit is None:
        size = data.size
        prob = data.flatten()
    else:
        size = limit
        prob = data[:limit].flatten()
    shape = data.shape[1]
    factor = tile(range(shape), size)
    doc = repeat(range(size), shape)
    df = pd.DataFrame(c_[prob, factor, doc],
                      columns=["prob", "factor", "doc"])

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="doc", hue="doc", col_wrap=5, size=1.5)

    # Draw a horizontal line to show the starting point
    grid.map(plt.axhline, y=0, ls=":", c=".5")

    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "factor", "prob", marker="o", ms=4)

    # Adjust the tick positions and labels
    grid.set(xticks=arange(shape), yticks=[-3, 3],
             xlim=(.0, shape), ylim=(0.0, 1.0))

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)

    grid.fig.suptitle(title)


facet_grid(predicted, limit=100, title="Predicted")
facet_grid(Y_test, limit=100, title="Actual")

sns.despine(left=True, bottom=True)

# print_predictions(predicted, document_X_title, selected_categories, idx_test, y=y_train_text, show_words=100, encode=False)

print('Loss: %1.4f' % score)
print('Accuracy: %1.4f' % acc)

plt.show()
