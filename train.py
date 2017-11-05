# coding=utf-8
from keras.callbacks import TensorBoard
from numpy import random
import os

import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer

from data_helper import word_to_vec, tokenize_documents, keras_prepare_text
from web_data import load_data_and_labels, data_folder_path

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

input_file = os.path.join(data_folder, 'cats.xlsx')
x, document_Y, y_train_text, df, selected_categories = load_data_and_labels(input_file)
document_X = df.Text

# X, Y = keras_prepare_text(df, y_train_text, max_sent_length=num_features, max_sents=document_max_num_words)
# number_of_documents = len(document_X)
# num_categories = len(selected_categories)

# Tokenized document collection
newsline_documents, number_of_documents = tokenize_documents(document_X, document_Y,
                                                             # lang='russian', regex=u'[А-Яа-яЁa-zA-Z^,!.\/+-=\']+')
                                                             lang='russian', regex=u'[А-Яа-яЁa-zA-Z\']+')

# x_x, y_y, num_categories_x = doc_to_vec(newsline_documents, number_of_documents, document_Y, selected_categories,
#                                         data_folder, model_name=doc2vec_model_name, num_features=num_features,
#                                         document_max_num_words=document_max_num_words)

# Create new Gensim Word2Vec model
X, Y, num_categories = word_to_vec(newsline_documents, number_of_documents, document_Y, selected_categories,
                                   data_folder, model_name=word2vec_model_name, num_features=num_features,
                                   document_max_num_words=document_max_num_words, sg=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0,
                         write_graph=True, write_images=True, write_grads=True)

model = Sequential()

model.add(LSTM(int(document_max_num_words * 1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, batch_size=128, epochs=15, validation_data=(X_test, Y_test), callbacks=[tbCallback])

model.save(data_folder + model_name)

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
