from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from statsmodels.tsa.ar_model import AutoReg


ACCENT_POSITIVE = "green"
ACCENT_NEGATIVE = "red"

HISTORY = {}
TODAY = datetime.today()


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url('dash-logo.png'),
        style={'float': 'right', 'height': 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")

    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


def generate_array(size=(4, 5)):
    return np.random.choice(["up", "down"], size, replace=True, p=[0.9, 0.1])


def make_card(status, device):
    color = "success"
    if status == "down":
        color = "danger"

    card = dbc.Card(
        [
            html.H3(status, className="card-title"),
            html.P(
                device,
                className="card-text",
                style={"font-size": "11pt"}
            )
        ],
        body=True,
        inverse=True,
        color=color,
        style={"margin": "15px"}
    )
    return card


def generate_pie(array):
    fig = go.Figure(
        data=go.Pie(
            labels=["up", "down"],
            values=[np.sum(array == "up"), np.sum(array == "down"),],
            marker={"colors": ["var(--accent_positive)", "var(--accent_negative)"]},
        )
    )
    fig.update_layout(paper_bgcolor="var(--background_content)")
    return fig


def generate_gauge(value):
    color = ACCENT_POSITIVE
    if value < 0.9:
        color = ACCENT_NEGATIVE
    fig = go.Figure(
        data=go.Indicator(
            mode="gauge+number",
            value=value,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Percentage of Devices Up"},
            gauge={
                "axis": {"range": [0, 1], "tickformat": "%"},
                "bar": {"color": color},
            },
            number={"valueformat": "%"},
        )
    )

    return fig


app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


charts_card = dbc.Card(
    [
        dbc.CardHeader('Percentage of Devices'),
        dbc.CardBody([
            dcc.Graph(id='pie-graph', style={'height': "calc(45vh - 50px)"}),
            dcc.Graph(id="forecast-graph", style={'height': "calc(45vh - 50px)"})
        ])
    ],
    style={}
)

card_block = dbc.Card(
    [
        dbc.CardHeader('Device Statuses'),
        dbc.CardBody(id='card-block')
    ],
    style={}
)

app.layout = dbc.Container(
    [
        dcc.Interval(id="update-interval", interval=3000, n_intervals=0),
        Header("IOT Ping Monitoring and Forecasting", app),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col([charts_card], md=4),
                dbc.Col(card_block, md=8)
            ]
        ),
    ],
    fluid=True,
)


@app.callback(
    [Output("card-block", "children"), Output("pie-graph", "figure"), Output('forecast-graph', 'figure')],
    [Input("update-interval", "n_intervals")],
)
def update_cards(n_intervals):
    array = generate_array()
    value = np.sum(array == "up") / array.size

    HISTORY[TODAY + timedelta(0, 3*n_intervals)] = value

    gauge = generate_gauge(value)
    cards = [
        dbc.Row(
            [dbc.Col(make_card(s, f"device {d}")) for d, s in enumerate(row)]
        ) for row in array
    ]


    # FORECASTING
    forecast_fig = go.Figure().update_layout(title='Not enough datapoints for forecasting')

    if len(HISTORY) >= 5:
        try:
            series = pd.Series(HISTORY).sort_index()
            print(series)
            model = AutoReg(series, lags=2, old_names=False)
            model_fit = model.fit()
            start = series.index.max()
            end = start + timedelta(0, 12)
            pred_df = model_fit.predict(start=start, end=end)

            forecast_fig = px.line(
                pred_df, 
                labels={'index': 'Future Timesteps', 'value': "Predicted %"},
                title="Forecasting with Univariate Autoregressive Processes")
            forecast_fig.update_layout(showlegend=False)
        except Exception as e:
            print(e)

    return cards, gauge, forecast_fig


if __name__ == "__main__":
    app.run_server(debug=True)
