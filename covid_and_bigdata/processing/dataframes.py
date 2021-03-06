import pandas
from datetime import datetime, date
from covid_and_bigdata.processing.language_complexity.text_statistics import TextStat
from readability import Readability


def monthly_complexity(sub_df: pandas.DataFrame, textstat: TextStat) -> pandas.DataFrame:
    """
    Grouping and aggregation helper

    Parameters
    ----------
    sub_df: `pandas.DataFrame`, required
        The sub_df after grouping

    textstat: `TextStat`, required
        The textstat object

    Returns
    ----------
    The result dataframe
    """
    concatenated_text = " . ".join(sub_df.cleaned_tweet.tolist())
    output_df = dict(month=[sub_df.iloc[0]['month']])
    output_df.update(textstat.analyze_text_complexity(concatenated_text))
    return pandas.DataFrame(output_df)


def daily_complexity(sub_df: pandas.DataFrame, textstat: TextStat) -> pandas.DataFrame:
    """
    Grouping and aggregation helper

    Parameters
    ----------
    sub_df: `pandas.DataFrame`, required
        The sub_df after grouping

    textstat: `TextStat`, required
        The textstat object

    Returns
    ----------
    The result dataframe
    """
    concatenated_text = " . ".join(sub_df.cleaned_tweet.tolist())
    output_df = dict(date=[sub_df.iloc[0]['date']])
    output_df.update(textstat.analyze_text_complexity(concatenated_text))
    return pandas.DataFrame(output_df)


def monthly_complexity_per_state(sub_df: pandas.DataFrame, textstat: TextStat) -> pandas.DataFrame:
    """
    Grouping and aggregation helper

    Parameters
    ----------
    sub_df: `pandas.DataFrame`, required
        The sub_df after grouping

    textstat: `TextStat`, required
        The textstat object

    Returns
    ----------
    The result dataframe
    """
    concatenated_text = " . ".join(sub_df.cleaned_tweet.tolist())
    output_df = dict(month=[sub_df.iloc[0]['month']], state=[sub_df.iloc[0]['state']])
    output_df.update(textstat.analyze_text_complexity(concatenated_text))
    return pandas.DataFrame(output_df)


def daily_complexity_per_state(sub_df: pandas.DataFrame, textstat: TextStat) -> pandas.DataFrame:
    """
    Grouping and aggregation helper

    Parameters
    ----------
    sub_df: `pandas.DataFrame`, required
        The sub_df after grouping

    textstat: `TextStat`, required
        The textstat object

    Returns
    ----------
    The result dataframe
    """
    concatenated_text = " . ".join(sub_df.cleaned_tweet.tolist())
    output_df = dict(date=[sub_df.iloc[0]['date']], state=[sub_df.iloc[0]['state']])
    output_df.update(textstat.analyze_text_complexity(concatenated_text))

    return pandas.DataFrame(output_df)


def get_dataframe_and_prepare(filepath: str) -> pandas.DataFrame:
    """
    Dataframe preparations

    Parameters
    ----------
    filepath: `str`, required
        The csv filepath

    Returns
    ----------
    The result dataframe
    """
    df = pandas.read_csv(filepath)
    df.drop(columns=[c for c in df.columns if c.startswith('Unnamed: ')], inplace=True)
    df['char_count'] = df.cleaned_tweet.copy().apply(lambda x: len(x) if isinstance(x, str) else 100000)
    df.timestamp = df.timestamp.apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S +0000 %Y').timestamp())
    df = df[df.char_count <= 280]
    df = df[~df.tweet.isna()]
    df = df.sort_values(by=['timestamp', 'state'])
    df['date'] = df['timestamp'].copy().apply(lambda x: datetime.fromtimestamp(int(x)).date())
    df['month'] = df['date'].copy().apply(lambda x: date(year=2020, month=x.month, day=1))
    return df
