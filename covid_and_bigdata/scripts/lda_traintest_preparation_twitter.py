import pandas
from tqdm import tqdm
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)
from datetime import datetime, date, timedelta
import pickle
from pandarallel import pandarallel
from sklearn.model_selection import GridSearchCV, train_test_split
from covid_and_bigdata.processing.lda_utils import *
warnings.simplefilter("ignore", DeprecationWarning)
pandarallel.initialize(progress_bar=True, nb_workers=10)

sns.set_style('whitegrid')


tqdm.pandas()
df = pandas.read_csv('df_with_symptoms_and_hate_score_and_language_complexity.csv')
df.drop(columns=[c for c in df.columns if c.startswith('Unnamed: ')], inplace=True)
df['char_count'] = df.cleaned_tweet.copy().apply(lambda x: len(x) if isinstance(x, str) else 100000)
df = df[df.char_count <= 280]
df = df.sort_values(by=['timestamp', 'state'])
df['date'] = df['timestamp'].copy().apply(lambda x: datetime.fromtimestamp(int(x)).date())
df['cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: re.sub('[,\.!?]', '', x))
df['cleaned_tweet'] = df['cleaned_tweet'].apply(lambda x: x.lower())
df['cleaned_tweet_lemmatized'] = df['cleaned_tweet'].progress_apply(lambda x: lemmatization(x))
train_usernames, test_usernames = train_test_split(df.username.unique().tolist(), test_size=0.2, shuffle=True)
train_df = df[df.username.isin(train_usernames)].copy()
test_df = df[df.username.isin(test_usernames)].copy()

with open('train_test_twitter.pkl', 'wb') as handle:
    pickle.dump({'train': train_df, 'test': test_df}, handle)