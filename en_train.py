# coding=utf-8
from numpy import random

import h5py

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM

from sklearn.model_selection import train_test_split

from data_helper import word_to_vec, tokenize_documents

from reuters import stream_reuters_documents, data_folder_path, get_category_data

# Set Numpy random seed
random.seed(1000)

# Newsline folder and format
data_folder = data_folder_path()


# Word2Vec number of features
num_features = 500
# Limit each newsline to a fixed number of words
document_max_num_words = 100
# Selected categories
selected_categories = ['pl_usa', 'to_cocoa', 'pl_uruguay', 'pl_albania', 'pl_barbados', 'pe_dunkel']

model_name = "reuters.model.h5"
word2vec_model_name = 'reuters.word2vec'

# Read all categories
category_data = get_category_data(data_folder)

document_X, document_Y, news_categories = stream_reuters_documents(category_data, columns=['Name', 'Type', 'Newslines'],
                                                                   selected_categories=selected_categories)


# Tokenized document collection
newsline_documents, number_of_documents = tokenize_documents(document_X, document_Y)

# Create new Gensim Word2Vec model
X, Y, num_categories = word_to_vec(newsline_documents, number_of_documents, document_Y,selected_categories,
                                   data_folder, model_name=word2vec_model_name, num_features=num_features,
                                   document_max_num_words=document_max_num_words)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()

model.add(LSTM(int(document_max_num_words*1.5), input_shape=(document_max_num_words, num_features)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, batch_size=128, epochs=5, validation_data=(X_test, Y_test))

model.save(data_folder + model_name)

# Evaluate model
score, acc = model.evaluate(X_test, Y_test, batch_size=128)

print('Score: %1.4f' % score)
print('Accuracy: %1.4f' % acc)
