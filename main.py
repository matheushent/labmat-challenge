import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from utils.process_data import text_process

data = pd.read_csv('dataset.csv')
data = data.drop(labels='Source.1', axis=1)
data.dropna(axis=0, inplace=True)
data['Business line'] = data['Business line'].replace({
    'Commercial': 'Comercial'
})

msg_train, msg_test, label_train, label_test = train_test_split(data['Summary'], data['Business line'], test_size=0.2)

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

        predictions = self.pipeline.predict(self.msg_test)
        print(classification_report(predictions, self.label_test))

model = BuildPipeline(msg_train, msg_test, label_train, label_test)
model.call()