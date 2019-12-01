from __future__ import print_function

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import string
import os

from optparse import OptionParser

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
import re

from utils.process_data import sort_coo, extract_topn_from_vector, text_process_with_lemmatization

parser = OptionParser()

parser.add_option("-o", "--option", dest="option", help="Which column to extract keywords. ('t': 'Title' or 's': 'Summary')")
(options, args) = parser.parse_args()

# nltk.download('stopwords')
# nltk.download('wordnet')

df = pd.read_csv('dataset.csv')

stop_words = set(stopwords.words('english'))

corpus = []
if options.option == 't':
    for i in range(df.shape[0]):
        text = text_process_with_lemmatization(df['Title'][i])
        text = " ".join(text)
        corpus.append(text)
elif options.option == 's':
    for i in range(df.shape[0]):
        text = text_process_with_lemmatization(df['Summary'][i])
        text = " ".join(text)
        corpus.append(text)
else:
    raise ValueError('You must pass a valid label via command line.')

cv = CountVectorizer(max_df=0.8 ,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
x = cv.fit_transform(corpus)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(x)

# the text to be processed
doc = corpus[477]
tf_idf_vector = tfidf_transformer.transform(cv.transform([doc]))

feature_names = cv.get_feature_names()

sorted_items = sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
 
# now print the results
print(f"\n{options.option}:")
print(doc)
print("\nKeywords:")
for k in keywords:
    print(k, keywords[k])