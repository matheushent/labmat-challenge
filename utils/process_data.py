from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string

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