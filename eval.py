# coding=utf-8
from numpy import random, arange, float32

import h5py, os

from sklearn.model_selection import train_test_split
from keras.models import load_model

from data_helper import print_predictions, tokenize_documents, doc_to_vec, RUSSIAN_REGEX
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
word2vec_model_name = 'cnsa.word2vec'
doc2vec_model_name = 'cnsa.doc2vec'

input_file = os.path.join(data_folder, 'webhose.csv')
x, document_Y, y_train_text, df, selected_categories = load_data_and_labels(input_file)
document_X = df.Text
document_X_title = df.Title

# Tokenized document collection
newsline_documents, number_of_documents = tokenize_documents(document_X, document_Y, decode=True,
                                                             lang='russian', regex=RUSSIAN_REGEX)

X, Y, num_categories = doc_to_vec(newsline_documents, number_of_documents, document_Y, selected_categories,
                                  data_folder, model_name=doc2vec_model_name, num_features=num_features,
                                  document_max_num_words=document_max_num_words)

indices = arange(Y.shape[0])
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.3)

model = load_model(data_folder + model_name)

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

predicted = model.predict_proba(X_test)

print_predictions(predicted, document_X_title, selected_categories, idx_test, y=y_train_text, show_words=100, encode=False)

print('Loss: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
