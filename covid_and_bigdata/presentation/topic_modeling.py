import numpy
import os
from pandarallel import pandarallel
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import warnings
import pickle
import gensim, spacy
import sklearn.decomposition
import sklearn.feature_extraction.text
import scipy.sparse
from typing import List

pandarallel.initialize(progress_bar=True)
warnings.simplefilter("ignore", DeprecationWarning)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
sns.set_style('whitegrid')


def plot_k_most_common_words(
        count_data: scipy.sparse.csr_matrix,
        count_vectorizer: sklearn.feature_extraction.text.CountVectorizer,
        k: int = 20) -> None:
    """
    This method helps in plotting the k top common words

    Parameters
    ----------
    count_data: `scipy.sparse.csr_matrix`, required
        The output of the count_vectorizer applied on the text
    count_vectorizer: `sklearn.feature_extraction.text.CountVectorizer`, required
        sklearn module for text features


    k: `int`, optional (default=20
        The number of the top words to be shown
    """
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts += t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[0:k]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='20 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# Helper function
def print_topics(model: sklearn.decomposition.LatentDirichletAllocation,
                 count_vectorizer: sklearn.feature_extraction.text.CountVectorizer,
                 n_top_words: int) -> None:
    """
    Printing the top words per topic

    Parameters
    -----------
    model: `sklearn.decomposition.LatentDirichletAllocation`, required
        sci-kit learn LDA model

    count_vectorizer: `sklearn.feature_extraction.text.CountVectorizer`, required
        text feature module

    n_top_words: `int`, required
        The number of top words per topic

    """
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))


def sent_to_words(sentence: str) -> List[str]:
    """
    Tokenizing sentence to words

    Parameters
    ----------
    sentence: `str`, required
        The sentence text

    Returns
    -----------
    List of tokens
    """
    return gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def lemmatization(sentence: str, allowed_postags: List[str] = ['NOUN', 'ADJ', 'VERB', 'ADV']) -> List[str]:
    """
    Parameters
    ----------
    sentence: `str`, required
        The sentence text

    allowed_postags: `List[str]`, optional (default=`['NOUN', 'ADJ', 'VERB', 'ADV']`)
        Allowed POS tags

    Returns
    -----------
    lemmatized tokens
    """
    sent_words = sent_to_words(sentence)
    doc = nlp(" ".join(sent_words))
    return " ".join(
        [token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags])


def plot_perplexity_of_train_and_test(lda_bundle_filepath: str, show_train: bool = False) -> go.Figure:
    """
    Parameters
    ----------
    lda_bundle_filepath: `str`, required
        The path to the checkpoint file of covision containing the lda models per number of components
        along with the performance values.

    show_train: `bool`, optional (default=False)
        If `True`, the perplexities for the original set (not just the held-out validation set)
        will be included in the final plot.

    Returns
    -----------
    It returns the plotly graph object figure of type `go.Figure`.
    """
    with open(lda_bundle_filepath, 'rb') as handle:
        lda_models_by_n_components = pickle.load(handle)
    perplexities = dict()
    perplexities['x'] = numpy.arange(2, 45)
    perplexities['train'] = numpy.array(
        [lda_models_by_n_components['n_components'][e]['train_data']['perplexity'] for e in range(2, 45)])
    perplexities['test'] = numpy.array(
        [lda_models_by_n_components['n_components'][e]['test_data']['perplexity'] for e in range(2, 45)])
    scores = dict()
    scores['x'] = numpy.arange(2, 45)
    scores['train'] = numpy.array(
        [lda_models_by_n_components['n_components'][e]['train_data']['log_likelihood'] for e in range(2, 45)])
    scores['test'] = numpy.array(
        [lda_models_by_n_components['n_components'][e]['test_data']['score'] for e in range(2, 45)])
    fig = go.Figure(layout=go.Layout(yaxis_title='Perplexity', xaxis_title='Number of Topics', template='plotly_white'))
    if show_train:
            fig.add_trace(go.Scatter(
            x=perplexities['x'],
            y=perplexities['train'],
            name='train',
            marker=dict(
                color='blue')
        ))
    fig.add_trace(go.Scatter(
        x=perplexities['x'],
        y=perplexities['test'],
        name='hold-out set',
        marker=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=[perplexities['x'][numpy.argmin(perplexities['test'])]],
        y=[numpy.min(perplexities['test'])],
        mode='markers',
        marker=dict(
            size=16,
            color='green',
            symbol='x'
        ),
        name='Minimum'
    ))
    return fig


def plot_perplexity_of_train_and_test_for_reddit(repo: str, show_train: bool = False) -> go.Figure:
    """
    Similar to the :func:`plot_perplexity_of_train_and_test`, for reddit.

    Parameters
    ----------
    lda_bundle_filepath: `str`, required
        The path to the checkpoint file of covision containing the lda models per number of components
        along with the performance values.

    show_train: `bool`, optional (default=False)
        If `True`, the perplexities for the original set (not just the held-out validation set)
        will be included in the final plot.

    Returns
    -----------
    It returns the plotly graph object figure of type `go.Figure`.
    """
    perplexities = dict()
    perplexities['x'] = numpy.arange(2, 50)
    perplexities['train'] = []
    perplexities['test'] = []
    for i in range(2, 50):
        with open(os.path.join(repo, '%dtopics.pkl' % i), 'rb') as handle:
            data = pickle.load(handle)
            perplexities['train'].append(data['perplexity'])
            perplexities['test'].append(data['test']['perplexity'])
    perplexities['train'] = numpy.array(
        perplexities['train'])
    perplexities['test'] = numpy.array(
        perplexities['test'])
    fig = go.Figure(layout=go.Layout(yaxis_title='Perplexity', xaxis_title='Number of Topics', template='plotly_white'))
    if show_train:
        fig.add_trace(go.Scatter(
            x=perplexities['x'],
            y=perplexities['train'],
            name='train',
            marker=dict(
                color='blue')
        ))
    fig.add_trace(go.Scatter(
        x=perplexities['x'],
        y=perplexities['test'],
        name='hold-out',
        marker=dict(color='green')
    ))

    fig.add_trace(go.Scatter(
        x=[perplexities['x'][numpy.argmin(perplexities['test'])]],
        y=[numpy.min(perplexities['test'])],
        mode='markers',
        marker=dict(
            size=16,
            color='green',
            symbol='x'
        ),
        name='Minimum'
    ))
    return fig
