#!/usr/bin/env python
# coding=utf-8

from web_data import multi_labels, clean_str
import pandas as pd
import numpy as np
from itertools import chain
import csv


def load_data_and_labels(filename):
    """Load sentences and labels"""

    def add_tuple(x1, x2):
        if x2 is np.nan:
            return tuple([str(x1)])
        else:
            return tuple([str(x1)]) + tuple(filter(lambda v: v is not 'nan', str(x2).split(";")))

    df = pd.read_csv(filename, parse_dates=True, quotechar='"', delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    df['Factor'] = df[['Factor1', 'Factor2']].apply(lambda (x1, x2): add_tuple(x1, x2), axis=1)
    selected = ['Factor', 'Text', 'Title']

    return load_data_and_labels_converter(df, selected)


def load_data_and_labels_converter(df, selected):
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # Drop non selected columns
    df = df.dropna(axis=0, how='any', subset=selected)  # Drop null rows
    df = df.reindex(np.random.permutation(df.index))  # Shuffle the dataframe

    # Map the actual labels to one hot labels
    labels = sorted(list(set(df[selected[0]].tolist())))
    labels = sorted(set(chain(*labels)))
    one_hot = np.zeros((len(labels), len(labels)), int)
    label_dict = dict(zip(labels, one_hot))

    def add_label(y):
        label = label_dict['0'].copy()
        for i in y:
            label[labels.index(i)] = 1
        return label

    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    pre_y_raw = df[selected[0]].apply(lambda y: add_label(y)).tolist()
    y_labels = multi_labels(pre_y_raw, labels)
    y_raw = dict(enumerate(pre_y_raw))
    return x_raw, y_raw, y_labels, df, labels
