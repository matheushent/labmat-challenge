from __future__ import print_function

from keras.layers import Dropout, Dense, GRU, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from optparse import OptionParser
import pandas as pd
import numpy as np
import os

from utils.process_data import load_data

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

def load_data_tokenizer(x_train, x_test, max_nb_words=1000, max_sequence_length=500):
    np.random.seed(11)
    text = np.concatenate((x_train, x_test), axis=0)
    text = np.array(text)
    tokenizer = Tokenizer(num_words=max_nb_words)
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    text = pad_sequences(sequences, maxlen=max_sequence_length)
    print('Found %s unique tokens.' % len(word_index))
    indices = np.arange(text.shape[0])
    # np.random.shuffle(indices)
    text = text[indices]
    print(text.shape)
    x_train = text[0:len(x_train), ]
    x_test = text[len(x_train):, ]
    embeddings_index = {}
    f = open("C:/Users/mathe/Desktop/Useful/glove.6B.50d.txt", encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except:
            pass
        embeddings_index[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embeddings_index))
    return (x_train, x_test, word_index, embeddings_index)

class RNN():
    def __init__(self, word_index, embeddings_index, nb_classes, max_sequence_length=500, embedding_dim=50):
        self.gru_node = 32
        self.layers = 3
        self.optimizer = Adam(lr=0.0005)
        self.model = self.build(word_index, embeddings_index, nb_classes, max_sequence_length, embedding_dim, dropout=0.3)

    def build(self, word_index, embeddings_index, nb_classes, max_sequence_length, embedding_dim, dropout=0.3):
        model = Sequential()
        embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))

        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found will be all zeros.
                if len(embedding_matrix[i]) != len(embedding_vector):
                    print('Could not broadcast input array from shape', str(len(embedding_matrix[i])), 'into shape', str(len(embedding_vector)), "MAKE SURE YOUR"
                                                                                                                                                 " EMBEDDING_DIM IS EQUAL TO EMBEDDING_VECTOR FILE, GloVe")
                    exit(1)
                embedding_matrix[i] = embedding_vector

        model.add(Embedding(
            len(word_index) + 1,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=max_sequence_length,
            trainable=True
        ))
        for i in range(self.layers):
            model.add(GRU(self.gru_node, return_sequences=True, recurrent_dropout=0.2))
            model.add(Dropout(dropout))

        model.add(GRU(self.gru_node, recurrent_dropout=0.2))
        model.add(Dropout(dropout))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(nb_classes, activation='softmax'))
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['accuracy']
        )

        return model

x_train, x_test, y_train, y_test = load_data(stratify)
x_train, x_test, word_index, embeddings_index = load_data_tokenizer(x_train, x_test)

enc = LabelEncoder().fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)

rnn = RNN(word_index, embeddings_index, nb_classes)
rnn.model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=20,
    batch_size=16,
    verbose=2,
    callbacks=[TensorBoard('./logs/rnn')]
)

rnn.model.save('./saved_models/rnn_2.hdf5', overwrite=True)

predictions = rnn.model.predict(x_test)
predictions = np.argmax(predictions, axis=1)
report = classification_report(enc.inverse_transform(y_test), enc.inverse_transform(predictions), output_dict=True)

df = pd.DataFrame(report).transpose()
out_path = os.path.join('reports', options.csv_name)
df.to_csv(out_path, encoding='utf-8')