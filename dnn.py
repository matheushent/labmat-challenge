from __future__ import print_function

from optparse import OptionParser
import pandas as pd
import numpy as np
import time
import os

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from utils.process_data import tfidf_vectorizer, load_data

parser = OptionParser()

parser.add_option("-n", "--name", dest="csv_name", help="Name of the csv file containing the classification report.")
parser.add_option("-s", "--stratify", dest="stratify", type=int, help="If use stratify in train_test_split or not. 1 to yes or 0 to don't. default=1", default=1)
(options, args) = parser.parse_args()

if not options.csv_name:
    parser.error('Error: csv file name must be specified. Pass -n via command line')

if options.stratify == 1:
    stratify = 1
else:
    stratify = None

nb_classes = 5

class DNN():
    def __init__(self, shape, n_classes):
        self.node = 512 # number of nodes in hidden layers
        self.layers = 4 # number of hidden layers
        self.optimizer = Adam(lr=0.0005)
        self.model = self.build(shape, n_classes)
        
    def build(self, shape, n_classes, dropout=0.3):
        model = Sequential()
        model.add(Dense(self.node, input_dim=shape, activation='relu'))
        model.add(Dropout(dropout))

        for i in range(self.layers):
            model.add(Dense(self.node, input_dim=self.node, activation='relu'))
            model.add(Dropout(dropout))
        
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model
    
x_train, x_test, y_train, y_test = load_data(stratify)
x_train, x_test = tfidf_vectorizer(x_train, x_test)

enc = LabelEncoder().fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

dnn = DNN(x_train.shape[1], nb_classes)
dnn.model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=13,
    batch_size=32,
    verbose=2
)

dnn.model.save('./saved_models/dnn.hdf5', overwrite=True)

predictions = dnn.model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
report = classification_report(y_test, predictions)

df = pd.DataFrame(report).transpose()
out_path = os.path.join('reports', options.csv_name)
df.to_csv(out_path, encoding='utf-8')