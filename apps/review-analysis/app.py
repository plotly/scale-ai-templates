from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import joblib
import pandas as pd
import numpy as np
import plotly.express as px


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")

    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


# Load datasets
old = pd.read_csv("old_reviews.csv")
new = pd.read_csv("new_reviews.csv")

old.month = pd.to_datetime(old.month)
new.month = pd.to_datetime(new.month)
drugs = pd.concat([old.drugName, new.drugName]).unique()
conditions = pd.concat([old.condition, new.condition]).unique()

# Load models
model = joblib.load("review_model/model.joblib")
vectorizer = joblib.load("review_model/vectorizer.joblib")
reducer = joblib.load("review_model/reducer.joblib")

# Create app
app = dash.Dash(external_stylesheets=[dbc.themes.LITERA])
server = app.server

controls = [
    dbc.FormGroup(
        [
            dbc.Label("Registered Date Range"),
            html.Br(),
            dcc.DatePickerRange(
                id="time-range",
                min_date_allowed=datetime(2008, 1, 1),
                max_date_allowed=datetime(2018, 1, 1),
                initial_visible_month=datetime(2017, 1, 1),
                start_date=datetime(2014, 1, 1).date(),
                end_date=datetime(2018, 1, 1).date(),
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Medical Condition"),
            dcc.Dropdown(
                id="medical-condition",
                value=conditions[0],
                options=[{"label": x, "value": x} for x in conditions],
                multi=True,
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Drug Name"),
            dcc.Dropdown(
                id="drug-name",
                value=drugs[0],
                options=[{"label": x, "value": x} for x in drugs],
                multi=True,
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Select Review"),
            html.Br(),
            dbc.ButtonGroup(
                [
                    dbc.Button(
                        "Prev", id="btn-prev", color="info", outline=True, n_clicks=0
                    ),
                    dbc.Button("Next", id="btn-next", color="info", n_clicks=0),
                ]
            ),
        ]
    ),
]

cards = [
    dbc.CardDeck(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Review ID"),
                    dbc.CardBody(html.H3(id="card-review-id", className="card-title")),
                ],
                color="warning",
                inverse=True,
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Medical Condition"),
                    dbc.CardBody(html.H3(id="card-condition", className="card-title")),
                ],
                color="danger",
                inverse=True,
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Drug Name"),
                    dbc.CardBody(html.H3(id="card-drug", className="card-title")),
                ],
                color="primary",
                inverse=True,
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Predicted Rating"),
                    dbc.CardBody(html.H2(id="card-prediction", className="card-title")),
                ],
                color="success",
                inverse=True,
            ),
        ]
    ),
    html.Br(),
    dbc.Card([dbc.CardHeader("Review Text"), dbc.CardBody(id="card-review-text")]),
    html.Br(),
    dbc.Card(
        [
            dbc.CardHeader("Prediction Confidence"),
            dbc.CardBody(dcc.Graph(id="prediction-confidence")),
        ]
    ),
]

app.layout = dbc.Container(
    [
        Header("Customer Review Rating Analysis", app),
        html.Hr(),
        dbc.Row([dbc.Col(dbc.Card(controls, body=True), md=4), dbc.Col(cards, md=8)]),
    ],
    fluid=True,
)


@app.callback(
    [Output("drug-name", "options"), Output("drug-name", "value")],
    [
        Input("medical-condition", "value"),
        Input("time-range", "start_date"),
        Input("time-range", "end_date"),
    ],
)
def update_drug_dropdown(condition, start_time, end_time):
    if type(condition) is str:
        condition = [condition]

    filtered = new[
        (new.month >= start_time)
        & (new.month <= end_time)
        & (new.condition.isin(condition))
    ]

    filtered_drugs = filtered.drugName.unique()
    new_options = [{"label": x, "value": x} for x in filtered_drugs]
    default = filtered_drugs[:4]

    return new_options, default


@app.callback(
    [
        Output("card-review-id", "children"),
        Output("card-condition", "children"),
        Output("card-drug", "children"),
        Output("card-review-text", "children"),
        Output("card-prediction", "children"),
        Output("prediction-confidence", "figure"),
    ],
    [
        Input("drug-name", "value"),
        Input("medical-condition", "value"),
        Input("time-range", "start_date"),
        Input("time-range", "end_date"),
        Input("btn-next", "n_clicks"),
        Input("btn-prev", "n_clicks"),
    ],
)
def update_cards(drug, condition, start_time, end_time, next_clicks, prev_clicks):
    if type(condition) is str:
        condition = [condition]

    if type(drug) is str:
        drug = [drug]

    if len(condition) == 0 or len(drug) == 0:
        return [dash.no_update] * 6

    filtered = new[
        (new.month >= start_time)
        & (new.month <= end_time)
        & new.drugName.isin(drug)
        & new.condition.isin(condition)
    ]

    filtered_ids = filtered.uniqueID.unique()

    ix = (next_clicks - prev_clicks) % filtered_ids.shape[0]
    selected = filtered.query(f"uniqueID == '{filtered_ids[ix]}'")

    # Run model now
    tf = vectorizer.transform(selected.review)
    vecs = reducer.transform(tf)
    pred_rating = model.predict(vecs)[0]
    probas = np.squeeze(model.predict_proba(vecs))

    fig = px.bar(
        x=model.classes_, y=probas, labels=dict(x="rating", y="confidence score")
    )

    return (
        selected.uniqueID,
        selected.condition,
        selected.drugName,
        selected.review,
        pred_rating,
        fig,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
