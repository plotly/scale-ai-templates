import os
import time

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    from https://github.com/facebook/prophet/issues/223

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def run_prophet(
    df,
    weekly_seasonality=False,
    start=2012,
    end=2016,
    uncertainty_samples=200,
    periods=180,
):
    yr = pd.to_datetime(df.ds).dt.year
    yr_df = df[(yr >= start) & (yr <= end)]
    with suppress_stdout_stderr():
        m = Prophet(
            weekly_seasonality=weekly_seasonality,
            uncertainty_samples=uncertainty_samples,
        )
        m.fit(yr_df)

    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    return m, forecast

def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            children
        ],
        **kwargs
    )

value2label = {
    "FOODS": "Business",
    "HOBBIES": "Other",
    "HOUSEHOLD": "Residential",
    "CA": "California",
    "TX": "Texas",
    "WI": "Wisconsin",
}

walmart = pd.read_csv("data.csv")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])
server = app.server  # expose server variable for Procfile


data_controls = [
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id="radio-category",
            options=[
                {"label": value2label[v], "value": v}
                for v in ["FOODS", "HOBBIES", "HOUSEHOLD"]
            ],
            value="FOODS",
        ),
        label="Category",
    ),
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id="radio-state",
            options=[{"label": value2label[v], "value": v} for v in ["CA", "TX", "WI"]],
            value="CA",
        ),
        label="State",
    ),
    NamedGroup(
        dcc.RangeSlider(
            id="range-years",
            min=2010,
            max=2016,
            step=1,
            value=[2012, 2014],
            marks={i: str(i) for i in range(2010, 2018, 2)},
        ),
        label="Years Fitted",
    ),
]

model_controls = [
    dbc.CardHeader("Model Controls"),
    dbc.CardBody(
        [
            NamedGroup(
                dbc.RadioItems(
            inline=True,
                    id="radio-seasonality",
                    options=[
                        {"label": "Enabled", "value": 1},
                        {"label": "Disabled", "value": 0},
                    ],
                    value=0,
                ),
                label="Weekly Seasonality",
            ),
            NamedGroup(
                dcc.Slider(
                    id="slider-forecast-days",
                    min=30,
                    max=365,
                    step=5,
                    value=180,
                    marks={i: str(i) for i in [30, 90, 180, 365]},
                ),
                label="Num Days Forecasted",
            ),
            NamedGroup(
                dcc.Slider(
                    id="slider-uncertainty-samples",
                    min=0,
                    max=600,
                    step=100,
                    value=200,
                    marks={i: str(i) for i in [0, 200, 400, 600]},
                ),
                label="Uncertainty Samples",
            ),
        ]
    )
]


app.layout = dbc.Container(
    [
        html.H1("Time Series Forecasting"),
        html.Hr(),
        dbc.Card(dbc.Row([dbc.Col(c) for c in data_controls]), body=True),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(model_controls), md=3),
                dbc.Col(dbc.Card(dcc.Graph(id="graph-forecast"), body=True), md=5),
                dbc.Col(dbc.Card(dcc.Graph(id="graph-components"), body=True), md=4),
            ]
        ),
    ],
    fluid=True
)


@app.callback(
    [Output("graph-forecast", "figure"), Output("graph-components", "figure")],
    [
        Input("radio-category", "value"),
        Input("radio-state", "value"),
        Input("radio-seasonality", "value"),
        Input("slider-forecast-days", "value"),
        Input("slider-uncertainty-samples", "value"),
        Input("range-years", "value"),
    ],
)
def run_forecast(category, state, seasonality, periods, uncertainty_samples, years):
    t0 = time.time()
    col = f"('{category}', '{state}')"
    df = walmart[["date", col]].rename(columns={"date": "ds", col: "y"})

    m, forecast = run_prophet(
        df,
        weekly_seasonality=seasonality,
        periods=periods,
        uncertainty_samples=uncertainty_samples,
        start=years[0],
        end=years[1],
    )

    # Plot figures
    margin = dict(l=30, r=30, t=80, b=30)

    fig_forecast = plot_plotly(m, forecast).update_layout(
        title="Forecasting with Prophet",
        width=None,
        height=None,
        margin=margin,
    )

    fig_components = plot_components_plotly(m, forecast).update_layout(
        title="Seasonal Components", width=None, height=None, margin=margin
    )

    t1 = time.time()
    print(f"Training and Inference time: {t1-t0:.2f}s.")

    return fig_forecast, fig_components


if __name__ == "__main__":
    app.run_server(debug=True)
