import pandas
import numpy
from pandarallel import pandarallel
from datetime import datetime, date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from scipy import signal
from typing import Dict, Tuple, List, Any

pandarallel.initialize(progress_bar=True)


def get_go_figure(
        tmp_df: pandas.DataFrame,
        events: List[Dict[str, Any]],
        half_window_days: int = 30,
        y: str = 'automated_readability_index',
        title: str = None,
        resolution: str = 'country',
        mode: str = 'markers+lines',
        line: Tuple[Dict[str, Any], Dict[str, Any]] = (dict(), dict(dash='dash')),
        font_size: int = 18
) -> go.Figure:
    """
    Parameters
    ----------
    tmp_df: `pandas.DataFrame`, required
        The dataframe used for the spatio-temporal trajectory plots. They should already be
        in the aggregate form, meaning that for the language complexity the values should have
        been computed after concatenation and elsewhere, and for other variables aggregations such as
        averaging must have been done by the caller prior to being passed to this method.

    events: `List[Dict[str, Any]]`, required
        The list of important news events to cross-reference. The first event in the
        list will be the center of attention.

    half_window_days: `int`, optional (default=30)
        The considered window will include 30 days before and 30 days after the first
        event given using the `events` variable.

    y: `str`, optional(default='automated_readability_index')
        The y axis variable name

    title: `str`, optional (default=None)
        If not none, the content will be used in the title of the figure.

    resolution: `str`, optional (default=`country`)
        The resolution of the trajectories, which is one of the options of `country` or `state`.

    mode: `str`, optional (default='markers+lines')
        plotly parameters

    line: `Tuple[Dict[str, Any], Dict[str, Any]]`, optional (default=`(dict(), dict(dash='dash'))`)
        The tuple of line argument for plotly plots for normal and smooth plots

    font_size: `int`, optional (default=18)
        Font size for plots

    Returns
    -----------
    It returns the plotly graph object figure of type `go.Figure`.
    """
    an_event = events[0]
    y_mapping = dict([
        ('<symptom>_count', 'Number of Symptoms in Tweet'),
        ('<covid_report>_count', 'Number of Potential Coronavirus Reports in Tweet'),
        ('<impact_body>_count', 'Potential Mention of Impact on Body'),
        ('<impact_activity>_count', 'Potential Mention of Impact on Activity'),
        ('hate_prob', 'CLAWS Task: Hate Probability'),
        ('counterhate_prob', 'CLAWS Task: Counter-hate Probability'),
        ('other_prob', 'CLAWS Task: Other Probability'),
        ('neutral_prob', 'CLAWS Task: Neutral Probability'),
        ('flesch_reading_ease', 'The Flesch Reading Ease formula'),
        ('flesch_kincaid_grade', 'The Flesch-Kincaid Grade Level'),
        ('gunning_fog', 'The Fog Scale'),
        ('smog_index', 'The SMOG Index'),
        ('automated_readability_index', 'Automatic Readibility Index'),
        ('coleman_liau_index', 'The Coleman-Liau Index'),
        ('linsear_write_formula', 'Linsear Write Formula'),
        ('dale_chall_readability_score', 'Dale-Chall Readability Score'),
    ])
    y = (y, y_mapping[y])

    # - well formatted dates
    event_date = [int(e) for e in an_event['date'].split('-')]
    event_date = date(year=event_date[0], month=event_date[1], day=event_date[2])
    start_date = event_date + timedelta(days=-half_window_days)
    end_date = event_date + timedelta(days=half_window_days)

    # - filtering the dates and making sure they are sorted
    tmp_df = tmp_df[(tmp_df.date >= start_date) & (tmp_df.date <= end_date)]
    tmp_df = tmp_df.sort_values(by='date')

    fig = go.Figure(layout=go.Layout(
        font=dict(size=font_size),
        title=go.layout.Title(
            text="Daily Average of {} around event: {} | Half Window: {} days".format(y[1], an_event['title'],
                                                                                      half_window_days) if title is None else title),
        xaxis_title="Date",
        yaxis_title=y[1],
        template='plotly_white'
    ))

    filtering_window = half_window_days // 2
    filtering_window = filtering_window if filtering_window % 2 == 1 else filtering_window - 1

    if resolution == 'country':
        fig.add_trace(
            go.Scatter(
                x=tmp_df['date'].to_numpy(),
                y=tmp_df[y[0]].to_numpy(),
                mode='lines',  # 'markers+lines',
                name='Daily Average',
                line=line[0]
            )
        )
        fig.add_trace(go.Scatter(
            x=tmp_df['date'].tolist(),
            y=signal.savgol_filter(tmp_df[y[0]].tolist(),
                                   filtering_window,  # window size used for filtering
                                   3),  # order of fitted polynomial
            mode=mode,
            marker=dict(
                size=6,
                color='mediumpurple',
                symbol='triangle-up'
            ),
            name='Daily Average (smooth)',
            line=line[1]
        ))
    elif resolution == 'state':
        fig = px.scatter(tmp_df, x='date', y=y[0], color='state')
        fig.update_layout(
            title_text="Daily Average of {} around event: {} | Half Window: {} days".format(y[1], an_event['title'],
                                                                                            half_window_days) if title is None else title,
            xaxis_title="Date",
            yaxis_title=y[1],
            template='plotly_white')
        for i in range(len(fig.data)):
            fig.data[i].update(mode=mode, line=line[0])

        for i in range(len(fig.data)):
            if filtering_window < fig.data[i]['y'].shape[0]:
                fig.add_trace(go.Scatter(
                    x=fig.data[i]['x'],
                    y=signal.savgol_filter(fig.data[i]['y'],
                                           filtering_window,  # window size used for filtering
                                           3),  # order of fitted polynomial
                    mode=mode,
                    marker=dict(
                        size=6,
                        color=fig.data[i]['marker']['color'],
                        symbol='triangle-up'
                    ),
                    name=fig.data[i]['name'] + '_smooth',
                    line=line[1]))

    else:
        raise ValueError

    # if an_event is not None:
    #     fig.add_trace(go.Scatter(
    #         x=tmp_df[tmp_df['date'] == event_date]['date'].tolist(),
    #         y=tmp_df[tmp_df['date'] == event_date][y[0]].tolist(),
    #         mode='markers+lines',
    #         marker=dict(
    #             size=16,
    #             color='red',
    #             symbol='x'
    #         ),
    #         name='{}'.format(an_event['title'])
    #     ))

    if events is not None:
        for an_event in events:
            event_date = [int(e) for e in an_event['date'].split('-')]
            event_date = date(year=event_date[0], month=event_date[1], day=event_date[2])
            y_event = numpy.linspace(tmp_df[y[0]].min() - 0.75 * tmp_df[y[0]].std(), tmp_df[y[0]].max() + 0.75 * tmp_df[y[0]].std())
            fig.add_trace(go.Scatter(
                x=[event_date] * len(y_event),
                y=y_event,
                mode='lines',
                marker=dict(
                    size=16,
                    symbol='x',
                    color='black'
                ),
                name='{}'.format(an_event['title']),
                showlegend=False
            ))
            fig.add_annotation(x=event_date, y=tmp_df[y[0]].max() + 0.4 * tmp_df[y[0]].std(),
                text=an_event['title'],
                showarrow=True,
                arrowhead=1)
    return fig
