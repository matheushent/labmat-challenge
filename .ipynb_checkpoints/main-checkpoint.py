import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('dataset.csv')
data = data.drop(labels='Source.1', axis=1)
data.dropna(axis=0, inplace=True)
print("We have {} words".format(data['Summary'].apply(lambda x: len(x.split(' '))).sum()))
data.Topic.value_counts().plot(kind='bar')
plt.show()