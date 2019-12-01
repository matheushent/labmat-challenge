from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from scipy.sparse import coo_matrix
import pandas as pd
import numpy as np
import string

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

lem = WordNetLemmatizer()

def text_process(summary):
    '''utility function to process the text
    # Arguments
        summary: the text to be processed
    
    # Returns
        the processed text
    '''

    # remove punctuation
    no_punc = [char for char in summary if char not in string.punctuation]

    # make the summary again
    no_punc = ''.join(no_punc)

    # remove stopwords
    summary = [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    
    return summary

def text_process_with_lemmatization(summary):
    '''utility function to process the text
    # Arguments
        summary: the text to be processed
    
    # Returns
        the processed text
    '''

    # remove punctuation
    no_punc = [char for char in summary if char not in string.punctuation]

    # make the summary again
    no_punc = ''.join(no_punc)

    # remove stopwords and lemmatization
    summary = [lem.lemmatize(word) for word in no_punc.split() if word.lower() not in stopwords.words('english')]
    
    return summary

def vectorization(x):
    '''utility function to vectorize the texts
    # Arguments
        x: message series
    
    # Returns
        the vectorized texts
    '''

    bow_transformer = CountVectorizer(analyzer=text_process).fit(x)
    messages_bow = bow_transformer.transform(x)

    print('Shape of sparse matrix: ', messages_bow.shape)
    sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
    print('Sparcity: {}'.format(sparsity))

    return messages_bow

def tfidf_transformer(x):
    '''utility function to make the tfidf transformation

    # Arguments
        x: message series

    # Returns
        the transformerd series
    '''

    msg = vectorization(x)
    transformer = TfidfTransformer().fit(msg)
    msg = transformer.transform(msg)

    return msg

def tfidf_vectorizer(x_train, x_test, max_nb_words=1000):
    '''utility function equivalent to CountVectorizer followed by TfidfTransformer

    # Arguments
        x_train: x data for training
        y_train: labels
        max_nb_words: maximum number of words

    # Returns
        the x_train and y_train transformed
    '''

    vectorizer_x = TfidfVectorizer(max_features=max_nb_words)
    x_train = vectorizer_x.fit_transform(x_train).toarray()
    x_test = vectorizer_x.transform(x_test).toarray()

    return (x_train, x_test)

def load_data(stratify):
    data = pd.read_csv('dataset.csv')
    data = data.drop(labels='Source.1', axis=1)
    data.dropna(axis=0, inplace=True)
    data = data[data['Business line'] != 'Agriculture']
    data['Business line'] = data['Business line'].replace({
        'Commercial': 'Comercial'
    })

    x = data['Summary']
    y = data['Business line']

    if stratify:
        stratify = y

    msg_train, msg_test, label_train, label_test = train_test_split(x, y, test_size=0.2, stratify=stratify)
    return msg_train, msg_test, label_train, label_test

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]
    
    return results