import numpy as np
import seaborn as sns
import warnings
import gensim, spacy, nltk, re
from sklearn.decomposition import LatentDirichletAllocation as LDA
from covid_and_bigdata.presentation.topic_modeling import plot_k_most_common_words, sent_to_words, lemmatization

warnings.simplefilter("ignore", DeprecationWarning)
sns.set_style('whitegrid')

nlp = spacy.load('en', disable=['parser', 'ner'])
