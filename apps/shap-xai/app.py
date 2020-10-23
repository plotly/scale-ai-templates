import json

import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, ALL
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def find_row_in_df(row, df):
    for ix in df.index:
        if np.all(df.iloc[ix].values == row):
            return ix

    return -1

def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            children
        ],
        **kwargs
    )



# Load Data
tips = px.data.tips()
tips["tip_pct"] = 100 * (tips.tip / tips.total_bill)


# Start the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI])
server = app.server  # expose server variable for Procfile

controls = [
    NamedGroup(
        dcc.Slider(
            id={"role": "model-input", "name": "total_bill"},
            min=0,
            max=50,
            value=tips.total_bill[0],
            step=1,
            marks={i: str(i) for i in range(0, 51, 10)},
        ),
        label="Total Bill ($)",
    ),
    NamedGroup(
        dcc.Slider(
            id={"role": "model-input", "name": "tip"},
            min=0,
            max=10,
            value=tips.tip[0],
            step=0.1,
            marks={i: str(i) for i in range(0, 11, 2)},
        ),
        label="Tip ($)",
        id="control-item-tip",
    ),
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id={"role": "model-input", "name": "sex"},
            options=[{"label": v, "value": v} for v in tips.sex.unique()],
            value=tips.sex[0],
            labelStyle={"display": "inline-block"},
        ),
        label="Sex",
        id="control-item-sex",
    ),
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id={"role": "model-input", "name": "smoker"},
            options=[{"label": v, "value": v} for v in tips.smoker.unique()],
            value=tips.smoker[0],
            labelStyle={"display": "inline-block"},
        ),
        label="Smoker",
    ),
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id={"role": "model-input", "name": "day"},
            options=[{"label": v, "value": v} for v in tips.day.unique()],
            value=tips.day[0],
            labelStyle={"display": "inline-block"},
        ),
        label="Weekday",
    ),
    NamedGroup(
        dbc.RadioItems(
            inline=True,
            id={"role": "model-input", "name": "time"},
            options=[{"label": v, "value": v} for v in tips.time.unique()],
            value=tips.time[0],
            labelStyle={"display": "inline-block"},
        ),
        label="Time",
    ),
    NamedGroup(
        dcc.Slider(
            id={"role": "model-input", "name": "size"},
            min=tips["size"].min(),
            max=tips["size"].max(),
            value=tips["size"][0],
            marks={
                i: str(i) for i in range(tips["size"].min(), tips["size"].max() + 1)
            },
        ),
        label="Size",
    ),
]


data_controls = dbc.Card(dbc.Row(
    [
        dbc.Col(c) for c in
        [
            dbc.Button("Random Train Sample", id="random-train", style={"margin-top": "20px"}, n_clicks=0),
            NamedGroup(
                dbc.RadioItems(
                inline=True,
                    id="radio-target",
                    options=[{"label": v, "value": v} for v in ["sex", "tip"]],
                    value="tip",
                ),
                label="Prediction Target",
            )
        ]
    ]),
    body=True
)


app.layout = dbc.Container(
    [
        html.H2("Tip estimation with SHAP"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dbc.Card([dbc.CardHeader("Model Input"), dbc.CardBody(controls)]), md=4),
                dbc.Col(
                    md=8,
                    children=[
                        data_controls,
                        html.Br(),
                        dbc.Card(
                            dcc.Graph(
                                id="model-output",
                                figure=go.Figure(),
                                style={"height": "calc(85vh - 150px)", "min-height": "500px"},
                            ),
                            body=True,
                        ),
                    ],
                )
            ]
        )
    ],
    fluid=True
)


@app.callback(
    [Output("control-item-sex", "style"), Output("control-item-tip", "style")],
    [Input("radio-target", "value")],
)
def hide_controls(target):
    if target == "tip":
        return {}, {"display": "none"}
    else:
        return {"display": "none"}, {}


# @app.callback(
#     [
#         Output({"role": "model-input", "name": "tip"}, 'disabled'),
#         Output({"role": "model-input", "name": "sex"}, 'options')
#     ],
#     [Input('radio-target', 'value')]
# )
# def disable_controls(target):
#     if target == 'tip':
#         return True, [{"label": v, "value": v, 'disabled': False} for v in tips.sex.unique()]
#     else:
#         return False, [{"label": v, "value": v, 'disabled': True} for v in tips.sex.unique()]


@app.callback(
    Output({"role": "model-input", "name": ALL}, "value"),
    [Input("random-train", "n_clicks")],
    [State({"role": "model-input", "name": ALL}, "id")],
)
def sample_training_data(n_clicks, input_ids):
    sample = tips.sample(n=1, random_state=n_clicks).iloc[0].to_dict()

    return [sample[d["name"]] for d in input_ids]


@app.callback(
    Output("model-output", "figure"),
    [
        Input({"role": "model-input", "name": ALL}, "value"),
        Input("radio-target", "value"),
    ],
    [State({"role": "model-input", "name": ALL}, "id")],
)
def interpret_model(input_list, target, input_ids):
    # Create dictionary based on the pattern-matched inputs
    inputs_dict = dict(zip([di["name"] for di in input_ids], input_list))

    if target == "tip":
        del inputs_dict["tip"]

        X = tips.drop(columns=["tip", "tip_pct"])
        y = tips["tip_pct"]

        cat_cols = ["sex", "smoker", "day", "time"]

        model = lgb.LGBMRegressor(n_jobs=1)

    else:  # target == 'sex'
        del inputs_dict["sex"]

        X = tips.drop(columns=["sex", "tip_pct"])
        y = tips["sex"]

        cat_cols = ["smoker", "day", "time"]

        model = RandomForestClassifier(n_jobs=1)

    # Encode strings into labels
    cat_classes = {}
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_classes[col] = le.classes_.tolist()
        encoders[col] = le

    # Fit Model
    model.fit(X, y)

    # Apply SHAP explainer
    explainer = shap.TreeExplainer(model)
    explainer.shap_values(X)

    # Create dataframe based on input dictionary
    inputs_df = pd.DataFrame([inputs_dict])

    for col in cat_cols:
        le = encoders[col]
        inputs_df[col] = le.transform(inputs_df[col])

    if target == "tip":
        base = explainer.expected_value
        shap_values = explainer.shap_values(inputs_df)[0]
    else:
        base = explainer.expected_value[1]
        shap_values = explainer.shap_values(inputs_df)[1][0]

    final = base + np.sum(shap_values)

    labels = [
        f"{col} = {cat_classes[col][int(val)]}"
        if col in cat_classes
        else f"{col} = {val:.3g}"
        for col, val in zip(inputs_df.columns, inputs_df.iloc[0, :])
    ]

    train_ix = find_row_in_df(row=inputs_df.values, df=X)

    x_fig = ["Base value"] + labels + ["Model output"]
    y_fig = [0] + shap_values.tolist() + [0]
    measure = ["absolute"] + ["relative"] * (len(labels) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Waterfall(
            orientation="v",
            measure=measure,
            x=x_fig,
            y=y_fig,
            text=[f"{value:.3g}" for value in y_fig],
            textposition="auto",
            hovertemplate="Moved by %{text}",
            hoverlabel_namelength=0,
            base=base,
            cliponaxis=False,
        )
    )

    fig.add_annotation(x=x_fig[0], y=base, text=f"Base = {base:.4g}", showarrow=False)

    fig.add_annotation(
        x=x_fig[-1], y=final, text=f"Output = {final:.4g}", showarrow=False
    )

    if target == "tip":
        fig.update_yaxes(range=[10, 24])
    else:
        fig.update_yaxes(range=[0, 1])

    fig.update_layout(uirevision=target, xaxis_fixedrange=True, modebar_orientation="h")

    if train_ix > -1 and target == "tip":
        fig.add_trace(
            go.Scatter(
                x=["Model output"],
                y=[y.loc[train_ix]],
                text=f"True value = {y.loc[train_ix]:.4g}%",
                mode="markers+text",
                textposition="top center",
            )
        )

    if target == "tip":
        fig.update_layout(
            title="Predicted Tips (%) explained by SHAP values", showlegend=False
        )
    elif train_ix > -1:
        fig.update_layout(
            title=f"Model Confidence that the client is male, explained by SHAP values (ground truth: {y.loc[train_ix].lower()})",
            showlegend=False,
        )
    else:
        fig.update_layout(
            title=f"Model Confidence that the client is male, explained by SHAP values",
            showlegend=False,
        )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
