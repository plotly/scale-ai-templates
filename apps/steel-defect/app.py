import os

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

from model import build_model, preprocess


def Header(name, app):
    title = html.H1(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("dash-logo.png"), style={"float": "right", "height": 60}
    )
    link = html.A(logo, href="https://plotly.com/dash/")

    return dbc.Row([dbc.Col(title, md=8), dbc.Col(link, md=4)])


# Load the segmentation model
model = build_model((256, 256, 1))
model._make_predict_function()
# graph = tf.get_default_graph()
model.load_weights("model.h5")

# Load steel defect images
neu_dir = os.walk("steel_images")
neu_samples = [os.path.join(r, f) for r, d, files in neu_dir for f in files]

img_map = {
    path: Image.open(path).convert("L").resize((256, 256)) for path in neu_samples
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

controls = [
    dbc.FormGroup(
        [
            dbc.Label("Select Image"),
            dbc.Select(
                id="selected-image",
                options=[
                    {"label": k.replace("steel_images/", ""), "value": k}
                    for k in neu_samples
                ],
                value=neu_samples[0],
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Defect Type"),
            dbc.RadioItems(
                id="defect-type",
                inline=True,
                options=[
                    {"label": k, "value": e}
                    for e, k in enumerate(["I", "II", "III", "IV"])
                ],
                value=0,
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Segmentation Display Mode"),
            dbc.Checklist(
                options=[{"label": "Overlay", "value": "overlay"}],
                id="switch-overlay",
                switch=True,
                value=[]
            ),
        ]
    ),
    dbc.FormGroup(
        [
            dbc.Label("Opacity"),
            dcc.Slider(id="opacity", min=0, max=1, step=0.05, value=0.5),
        ],
    ),
]

app.layout = dbc.Container(
    [
        Header("Steel Defect Segmentation", app),
        html.Hr(),
        dbc.Row([dbc.Col(c) for c in controls]),
        dbc.Row(
            [
                dbc.Col(dbc.Card(dcc.Graph(id="original-image"), body=True)),
                dbc.Col(dbc.Card(dcc.Graph(id="defect-segments"), body=True)),
            ]
        ),
    ],
    fluid=True,
)

@app.callback(
    [
        Output('original-image', 'figure'),
        Output('defect-segments', 'figure')
    ],
    [
        Input('selected-image', 'value'),
        Input('defect-type', 'value'),
        Input('switch-overlay', 'value'),
        Input('opacity', 'value')
    ]
)
def run_segmentation(img_name, defect_type, switches, opacity):
    # Retrieve image and run model
    im = img_map[img_name]
    defects = model.predict(preprocess(im), batch_size=1)[0]

    # Generate the figures
    fig_segment = px.imshow(defects[:, :, defect_type], title='Segmentation Map (U-Net)')
    fig_im = px.imshow(im, color_continuous_scale='gray', title='Original image')
    fig_im.update_layout(coloraxis_showscale=False)

    if 'overlay' in switches:
        fig_im.add_trace(go.Heatmap(z=defects[:, :, defect_type], opacity=opacity))
        fig_im.update_layout(title='Original image (with overlay)')


    return fig_im, fig_segment


if __name__ == "__main__":
    app.run_server(debug=True)
