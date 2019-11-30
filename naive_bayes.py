from __future__ import print_function

from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd
import time
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from utils.process_data import text_process

parser = OptionParser()

parser.add_option("-n", "--name", dest="csv_name", help="Name of the csv file containing the classification report.")
parser.add_option("-s", "--stratify", dest="stratify", type=int, help="If use stratify in train_test_split or not. 1 to yes or 0 to don't. default=1", default=1)
(options, args) = parser.parse_args()

if not options.csv_name:
    parser.error('Error: csv file name must be specified. Pass -n via command line')

init = time.time()

data = pd.read_csv('dataset.csv')
data = data.drop(labels='Source.1', axis=1)
data.dropna(axis=0, inplace=True)
data = data[data['Business line'] != 'Agriculture']
data['Business line'] = data['Business line'].replace({
    'Commercial': 'Comercial'
})

x = data['Summary']
y = data['Business line']

if options.stratify == 1:
    stratify = y
else:
    stratify = None

msg_train, msg_test, label_train, label_test = train_test_split(x, y, test_size=0.2, stratify=stratify)

class BuildPipeline():
    def __init__(self, msg_train, msg_test, label_train, label_test):
        self.msg_train = msg_train
        self.msg_test = msg_test
        self.label_train = label_train
        self.label_test = label_test
        self.pipeline = Pipeline([
            ('vectorizer', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])
    def call(self):
        self.pipeline.fit(self.msg_train, self.label_train)

        self.predictions = self.pipeline.predict(self.msg_test)
        return classification_report(self.predictions, self.label_test, output_dict=True)

model = BuildPipeline(msg_train, msg_test, label_train, label_test)
report = model.call()

end = time.time()
print('The entire process took {} seconds'.format(end - init))

df = pd.DataFrame(report).transpose()
out_path = os.path.join('reports', options.csv_name)
df.to_csv(out_path, encoding='utf-8')