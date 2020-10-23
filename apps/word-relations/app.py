import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_cytoscape as cyto
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            children
        ],
        **kwargs
    )


def find_neighbors(vec, n=8, offset=1, method="cosine"):
    if method == "cosine":
        scores = cosine_similarity(vec.reshape(1, -1), w2v_mat).squeeze()
        neighbors = scores.argsort()[::-1][offset : n + offset]
    else:
        scores = np.linalg.norm(w2v_mat - vec, axis=-1)
        neighbors = scores.argsort()[offset : n + offset]

    neighbor_scores = scores[neighbors]
    neighbor_words = w2v_vocab[neighbors]

    return neighbor_words, neighbor_scores


BLUE = "#0E34A0"
GRAY = "#9c9c9c"
RED = "#FB3640"
YELLOW = "#BB9F06"


base_stylesheet = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": BLUE,
            "text-background-color": BLUE,
            "color": "white",
            "text-background-opacity": 1,
            "text-valign": "center",
            "width": 35,
            "height": 35,
        },
    },
    {"selector": "edge", "style": {"line-color": BLUE}},
    {
        "selector": ".neighbors",
        "style": {
            "background-color": RED,
            "line-color": RED,
            "color": RED,
            "text-background-opacity": 0,
        },
    },
    {
        "selector": "node.neighbors",
        "style": {"width": 20, "height": 20, "text-valign": "top",},
    },
    {
        "selector": 'edge[target = "plus"], edge[target = "minus"], edge[target = "final"]',
        "style": {
            "mid-target-arrow-shape": "triangle",
            "mid-target-arrow-color": BLUE,
            "arrow-scale": 1.5,
        },
    },
]


# Load COLA layout
cyto.load_extra_layouts()

# Load glove
w2v_mat = np.load("w2v_mat.npy")
w2v_vocab = np.load("w2v_vocab.npy")
w2v_vocab_set = set(w2v_vocab)
w2v_dict = {w2v_vocab[i]: w2v_mat[i] for i in range(w2v_vocab.shape[0])}
print(f"Word2vec Loaded for {len(w2v_vocab)} words")


# Define cytoscape graph
cytoscape_component = cyto.Cytoscape(
    elements=[],
    id="cytoscape",
    layout={"name": "cola", "animate": True},
    style={"width": "100%", "height": "calc(75vh - 120px)"},
)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server  # expose server variable for Procfile

controls = [
    NamedGroup(
        dbc.Select(
            id="dropdown-preset",
            options=[
                {"label": s, "value": s.replace("- ", "").replace("+ ", "")}
                for s in [
                    "france - paris + tokyo",
                    "king - man + woman",
                    "california - state + country",
                    "sister - brother + grandson",
                    "bigger - big + cold",
                ]
            ],
            value="france paris tokyo",
        ),
        label="Preset",
    ),
    NamedGroup(
        dbc.Input(id="input-start", type="text", value="france"), label="Start Word",
    ),
    NamedGroup(
        dbc.Input(id="input-minus", type="text", value="paris"),
        label="Substract this word",
    ),
    NamedGroup(
        dbc.Input(id="input-plus", type="text", value="tokyo"), label="Add this word",
    ),
    dbc.Button("Run", id="button-run", color='primary', n_clicks=0, style={"margin-top": '28px'}),
    html.I(id="error-output"),
]


app.layout = dbc.Container(
    [
        html.H2("Word2Vec Embeddings Spatial relations"),
        html.Hr(),
        dbc.Card(dbc.Row([dbc.Col(c) for c in controls]), body=True),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(dbc.Card(cytoscape_component, body=True), md=7),
                dbc.Col(dcc.Graph(id="graph-distance"), md=5),
            ]
        ),
    ],
    fluid=True
)


@app.callback(
    [Output("button-run", "disabled"), Output("error-output", "children")],
    [
        Input("input-start", "value"),
        Input("input-plus", "value"),
        Input("input-minus", "value"),
    ],
)
def validate_inputs(start, plus, minus):
    for word in [start, plus, minus]:
        if word not in w2v_vocab_set:
            error_message = (
                f'The word "{word}" is not in the vocabulary. Please change it.'
            )

            if word == "":
                error_message = "Input is empty. Please write something."

            return True, error_message
    return False, None


@app.callback(
    Output("graph-distance", "figure"),
    [Input("cytoscape", "tapNodeData"), Input("button-run", "n_clicks")],
    [
        State("input-start", "value"),
        State("input-plus", "value"),
        State("input-minus", "value"),
    ],
)
def update_bar(node, n_clicks, start, plus, minus):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None

    if ctx.triggered[0]["prop_id"] == "button-run.n_clicks":
        node = {"id": "final", "label": f"{start} - {minus} + {plus}"}

    if node is None:
        return px.bar(title="Please tap on a node")

    vals = node["label"].split()
    offset = 0

    if "neighbor" in node["id"] or "start" == node["id"]:
        offset = 1
        vec = w2v_dict[vals[0]]

    elif "final" == node["id"]:
        v1, op, v2, op2, v3 = vals
        vec = w2v_dict[v1] - w2v_dict[v2] + w2v_dict[v3]

    elif "plus" == node["id"]:
        v1, op, v2 = vals
        vec = w2v_dict[v1] + w2v_dict[v2]

    else:
        v1, op, v2 = vals
        vec = w2v_dict[v1] - w2v_dict[v2]

    neighbor_words, weights = find_neighbors(vec, offset=offset, n=10)
    first_neighbor = [n for n in neighbor_words if n not in [start, plus, minus]][0]

    cmap = {start: GRAY, plus: GRAY, minus: GRAY, first_neighbor: YELLOW}

    fig = px.bar(
        y=neighbor_words[::-1],
        x=weights[::-1],
        orientation="h",
        title=f'Nearest Neighbors of {node["label"]}',
        labels={"x": "Cosine Similarity", "y": "Neighbors"},
        template='plotly_dark'
    )

    fig.update_traces(marker_color=[cmap.get(n, RED) for n in neighbor_words[::-1]])

    return fig


@app.callback(
    [
        Output("input-start", "value"),
        Output("input-minus", "value"),
        Output("input-plus", "value"),
        Output("button-run", "n_clicks"),
    ],
    [Input("dropdown-preset", "value")],
    [State("button-run", "n_clicks")],
)
def update_inputs(preset, n_clicks):
    return preset.split() + [n_clicks + 1]


@app.callback(
    [Output("cytoscape", "elements"), Output("cytoscape", "stylesheet")],
    [Input("button-run", "n_clicks")],
    [
        State("input-start", "value"),
        State("input-plus", "value"),
        State("input-minus", "value"),
    ],
)
def compute_arithmetic(n_clicks, start, plus, minus):
    stylesheet = base_stylesheet

    final_vec = w2v_dict[start] - w2v_dict[minus] + w2v_dict[plus]
    plus_vec = w2v_dict[start] + w2v_dict[plus]
    minus_vec = w2v_dict[start] - w2v_dict[minus]

    all_neighbors = {
        "start": find_neighbors(w2v_dict[start], offset=1),
        "plus": find_neighbors(plus_vec, offset=0),
        "minus": find_neighbors(minus_vec, offset=0),
        "final": find_neighbors(final_vec, offset=0),
    }

    elements = [
        {"data": {"id": "start", "label": start}},
        {"data": {"id": "plus", "label": f"{start} + {plus}"}},
        {"data": {"id": "minus", "label": f"{start} - {minus}"}},
        {"data": {"id": "final", "label": f"{start} - {minus} + {plus}"}},
        {"data": {"source": "start", "target": "plus", "label": f"+ {plus}"}},
        {"data": {"source": "start", "target": "minus", "label": f"- {minus}"}},
        {"data": {"source": "plus", "target": "final", "label": f"- {minus}"}},
        {"data": {"source": "minus", "target": "final", "label": f"+ {minus}"}},
    ]

    for main, (neighbors, weights) in all_neighbors.items():
        first_neighbor = [n for n in neighbors if n not in [start, plus, minus]][0]

        for n, neighb in enumerate(neighbors):
            elements.append(
                {
                    "data": {"id": f"{main}-neighbor-{n}", "label": neighb},
                    "classes": "neighbors",
                }
            )
            elements.append(
                {
                    "data": {
                        "source": main,
                        "target": f"{main}-neighbor-{n}",
                        "weight": weights[n],
                    },
                    "classes": "neighbors",
                }
            )

            if neighb in [start, plus, minus]:
                stylesheet += [
                    {
                        "selector": f'node.neighbors[label = "{neighb}"]',
                        "style": {"background-color": GRAY, "color": GRAY},
                    },
                    {
                        "selector": f'edge[target = "{main}-neighbor-{n}"]',
                        "style": {"line-color": GRAY},
                    },
                ]

            if neighb == first_neighbor:
                stylesheet += [
                    {
                        "selector": f'node[label = "{first_neighbor}"][id = "{main}-neighbor-{n}"]',
                        "style": {
                            "background-color": YELLOW,
                            "color": YELLOW,
                            "shape": "star",
                        },
                    },
                    {
                        "selector": f'edge[target = "{main}-neighbor-{n}"]',
                        "style": {"line-color": YELLOW},
                    },
                ]

    return elements[::-1], stylesheet


if __name__ == "__main__":
    app.run_server(debug=True)
