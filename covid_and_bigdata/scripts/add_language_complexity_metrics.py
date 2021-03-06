import pandas
import numpy
import os
from tqdm import tqdm
from pandarallel import pandarallel
from readability import Readability
from covid_and_bigdata.processing.dataframes import *
from covid_and_bigdata.processing.utilities import *
import argparse
from covid_and_bigdata.processing.language_complexity.text_statistics import TextStat

pandarallel.initialize(progress_bar=True)
tqdm.pandas()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="the input csv file of the twitter dataset")
    parser.add_argument("--output", type=str, help="the output repository")
    parser.add_argument("--aggregate", action='store_true')
    args = parser.parse_args()
    print("loading data...\n")
    df = pandas.read_csv(os.path.abspath(args.input))

    print("preprocessing...\n")
    df.timestamp = df.timestamp.progress_apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y').timestamp())
    df['month'] = df.timestamp.copy().progress_apply(change_timestamp_to_month)
    df['date'] = df['timestamp'].copy().progress_apply(lambda x: datetime.fromtimestamp(x).date())
    df = df[df.cleaned_tweet.progress_apply(lambda x: isinstance(x, str))]
    df['character_length'] = df['cleaned_tweet'].copy().progress_apply(lambda x: len(x))
    df = df[df['character_length'] < 300]

    textstat = TextStat()

    if args.aggregate:
        print("per country - daily...\n")
        df_daily = df.copy().groupby('date').parallel_apply(lambda x: daily_complexity(x, textstat))
        df_daily.to_csv(os.path.join(os.path.abspath(args.output), 'df_daily.csv'))
        print("per country - monthly...\n")
        df_monthly = df.copy().groupby('month').parallel_apply(lambda x: monthly_complexity(x, textstat))
        df_monthly.to_csv(os.path.join(os.path.abspath(args.output), 'df_monthly.csv'))
        print("per state - daily...\n")
        df_daily_per_state = df.copy().groupby(['state', 'date']).parallel_apply(lambda x: daily_complexity_per_state(x, textstat))
        df_daily_per_state.to_csv(os.path.join(os.path.abspath(args.output), 'df_daily_per_state.csv'))
        print("per state - monthly...\n")
        df_monthly_per_state = df.copy().groupby(['state', 'month']).parallel_apply(lambda x: monthly_complexity_per_state(x, textstat))
        df_monthly_per_state.to_csv(os.path.join(os.path.abspath(args.output), 'df_monthly_per_state.csv'))
    else:
        for met in textstat.meta.keys():
            for attr in textstat.meta[met]:
                key = "_".join([met, attr])
                print("computing {}...\n".format(key))
                df[key] = df.cleaned_tweet.parallel_apply(lambda x: textstat.analyze_text_by_single_method(text=x, method=met, attribute=attr))

    print("all finished.\n")
