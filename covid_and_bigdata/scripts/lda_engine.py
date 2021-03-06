import re
import argparse
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy
import pickle
import gc
import os
import pandas
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from tqdm import tqdm


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from nltk.corpus import stopwords


def get_corpus(df: pandas.DataFrame, text_column: str, id2word=None):
    data = df[text_column].tolist()
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", "", sent) for sent in data]
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # See trigram example
    # print(trigram_mod[bigram_mod[data_words[0]]])

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Create Dictionary
    if id2word is None:
        id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
    texts = data_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    return corpus, id2word, data_lemmatized


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="path to the pickle file containing train and test dataframe",
                        required=True)
    parser.add_argument("--text_column", type=str, help="the column name corresponding to the text values",
                        required=True)
    parser.add_argument("--topic_range", type=str, help="min topic and max topic entered as 2:6",
                        required=True)
    parser.add_argument("--output_repo", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    paths = {
        'checkpoints': os.path.join(os.path.abspath(args.output_repo), 'checkpoints'),
        'visualizations': os.path.join(os.path.abspath(args.output_repo), 'visualizations'),
        'reports': os.path.join(os.path.abspath(args.output_repo), 'reports')
    }

    for key in paths:
        os.makedirs(paths[key], exist_ok=True)

    with open(os.path.abspath(args.dataset), 'rb') as handle:
        datasets = pickle.load(handle)
        train_df = datasets['train'].copy()
        del datasets['train']
        test_df = datasets['test'].copy()
        del datasets['test']
        gc.collect()

    train_df = train_df[train_df[args.text_column].apply(lambda x: isinstance(x, str))]
    test_df = test_df[test_df[args.text_column].apply(lambda x: isinstance(x, str))]

    print(">> (status): building the train corpus...\n")
    train_corpus, id2word, train_data_lemmatized = get_corpus(train_df, args.text_column)

    print(">> (status): building the test corpus...\n")
    test_corpus, _, test_data_lemmatized = get_corpus(test_df, args.text_column, id2word=id2word)

    results_df = {
        "mode": [], "log_perplexity": [], "coherence": [], "n_topics": []
    }

    min_n, max_n = [int(e) for e in args.topic_range.split(':')]
    for n_topics in tqdm(range(min_n, max_n + 1)):
        print("-" * 10)
        print(f" => analyzing {n_topics} topics...\n")
        print(">> (status): training the LDA model...\n")
        lda_model = gensim.models.ldamulticore.LdaMulticore(
            corpus=train_corpus,
            id2word=id2word,
            num_topics=n_topics,
            random_state=131,
            chunksize=10000,
            passes=5,
            per_word_topics=True,
            workers=args.workers
        )
        # ---
        lda_model.save(os.path.join(paths['checkpoints'], '%dtopics.model' % n_topics))

        # -- train
        results_df['n_topics'].append(n_topics)
        results_df['mode'].append('train')
        train_log_perplexity = lda_model.log_perplexity(train_corpus)
        print(f"\ttrain_log_perplexity: {train_log_perplexity:.5f}\n")
        results_df['log_perplexity'].append(train_log_perplexity)

        coherence_model_lda = CoherenceModel(
            model=lda_model,
            texts=train_data_lemmatized,
            dictionary=id2word,
            coherence='c_v')
        train_coherence = coherence_model_lda.get_coherence()
        print(f"\ttrain_coherence: {train_coherence:.5f}\n")
        results_df['coherence'].append(train_coherence)
        vis = pyLDAvis.gensim.prepare(lda_model, train_corpus, id2word)
        pyLDAvis.save_html(vis, os.path.join(paths['visualizations'], '%dtopics_train.html' % n_topics))


        # -- test
        results_df['n_topics'].append(n_topics)
        results_df['mode'].append('test')
        test_log_perplexity = lda_model.log_perplexity(test_corpus)
        print(f"\ttest_log_perplexity: {test_log_perplexity:.5f}\n")
        results_df['log_perplexity'].append(test_log_perplexity)

        coherence_model_lda = CoherenceModel(
            model=lda_model,
            texts=test_data_lemmatized,
            dictionary=id2word,
            coherence='c_v')
        test_coherence = coherence_model_lda.get_coherence()
        print(f"\ttest_coherence: {train_coherence:.5f}\n")
        results_df['coherence'].append(test_coherence)

        vis = pyLDAvis.gensim.prepare(lda_model, train_corpus, id2word)
        pyLDAvis.save_html(vis, os.path.join(paths['visualizations'], '%dtopics_test.html' % n_topics))

        results_df = pandas.DataFrame(results_df)
        try:
            results_df.to_csv(os.path.join(paths['reports'], 'results.csv'))
        except:
            import pdb

            pdb.set_trace()


