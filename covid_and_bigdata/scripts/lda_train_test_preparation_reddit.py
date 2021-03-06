from covid_and_bigdata.processing.lda_utils import *
from io import StringIO
import pandas
import numpy
import os
import gc
import re
import pickle, json
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
import warnings
from pyLDAvis import sklearn as sklearn_lda
import pickle
import pyLDAvis
from pandarallel import pandarallel
from wordcloud import WordCloud
from sklearn.model_selection import GridSearchCV, train_test_split
import gensim, spacy, nltk, re
from sklearn.decomposition import LatentDirichletAllocation as LDA

warnings.simplefilter("ignore", DeprecationWarning)
pandarallel.initialize(progress_bar=True, nb_workers=10)
sns.set_style('whitegrid')

nlp = spacy.load('en', disable=['parser', 'ner'])


tqdm.pandas()
df = pandas.read_csv('reddit_tagged_withcounts.csv')
df = df[df.text.apply(lambda x: isinstance(x, str))]
df.drop(columns=[c for c in df.columns if c.startswith('Unnamed: ')], inplace=True)
df['cleaned_text'] = df['text'].copy().apply(lambda x: re.sub('[,\.!?]', '', x))
df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x.lower())
df['cleaned_text_lemmatized'] = df['cleaned_text'].progress_apply(lambda x: lemmatization(x))
train_usernames, test_usernames = train_test_split(df.user_id.unique().tolist(), test_size=0.2, shuffle=True,
                                                   random_state=23)
train_df = df[df.user_id.isin(train_usernames)].copy()
test_df = df[df.user_id.isin(test_usernames)].copy()

with open('train_test_reddit.pkl', 'wb') as handle:
    pickle.dump({'train': train_df, 'test': test_df}, handle)
