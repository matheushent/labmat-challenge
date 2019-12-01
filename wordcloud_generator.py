from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

from optparse import OptionParser

from nltk.corpus import stopwords

from utils.process_data import text_process_with_lemmatization

parser = OptionParser()

parser.add_option("-o", "--option", dest="option", help="Which column to extract keywords. ('t': 'Title' or 's': 'Summary')")
(options, args) = parser.parse_args()

df = pd.read_csv('dataset.csv')

stop_words = set(stopwords.words('english'))

corpus = []
if options.option == 't':
    name = 'title'
    for i in range(df.shape[0]):
        text = text_process_with_lemmatization(df['Title'][i])
        text = " ".join(text)
        corpus.append(text)
elif options.option == 's':
    name = 'summary'
    for i in range(df.shape[0]):
        text = text_process_with_lemmatization(df['Summary'][i])
        text = " ".join(text)
        corpus.append(text)
else:
    raise ValueError('You must pass a valid label via command line.')
    
wordcloud = WordCloud(background_color=None,
                      stopwords=stop_words,
                      max_words=100,
                      max_font_size=50, 
                      random_state=42,
                      mode='RGBA',
                      colormap='tab20c'
                      ).generate(str(corpus))

fig = plt.figure(1)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
wordcloud.to_file(name + '_wordcloud.png')
# fig.savefig(name + '_wordcloud.png', transparent=True)