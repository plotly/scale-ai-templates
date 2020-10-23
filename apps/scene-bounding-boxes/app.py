import base64
from io import BytesIO
import os
import time
import urllib

import cv2
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask_caching import Cache
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests


def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            children
        ],
        **kwargs
    )


def array_to_b64(img, enc="jpg"):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    is_success, buffer = cv2.imencode(f".{enc}", img)
    io_buf = BytesIO(buffer)
    encoded = base64.b64encode(io_buf.getvalue()).decode("utf-8")
    return f"data:img/{enc};base64, " + encoded


def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists.")
    else:
        print(f"{filename} does not exist. Downloading...", end=" ")
        r = requests.get(url, allow_redirects=True)
        with open(filename, "wb") as f:
            f.write(r.content)
        print("Done.")


def get_selected_frames(summary, label, min_elts, max_elts):
    return summary[
        np.logical_and(summary[label] >= min_elts, summary[label] <= max_elts)
    ].index


def load_image(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
    return image


def load_network(config_path, weights_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    output_layer_names = net.getLayerNames()
    output_layer_names = [
        output_layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()
    ]
    return net, output_layer_names


def add_bbox(fig, x0, y0, x1, y1, color="red", opacity=0.5, name=""):
    fig.add_scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode="lines",
        fill="toself",
        opacity=opacity,
        marker_color=color,
        hoveron="fills",
        hoverlabel_namelength=-1,
        name=name,
    )


def create_summary(metadata):
    one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
    summary = (
        one_hot_encoded.groupby(["frame"])
        .sum()
        .rename(
            columns={
                "label_biker": "biker",
                "label_car": "car",
                "label_pedestrian": "pedestrian",
                "label_trafficLight": "traffic light",
                "label_truck": "truck",
            }
        )
    )
    return summary


def yolo_v3(image, net, output_layer_names, confidence_threshold, overlap_threshold):
    # Run the YOLO neural net.
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layer_names)

    # Supress detections in case of too low confidence or too much overlap.
    boxes, confidences, class_IDs = [], [], []
    H, W = image.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box.astype("int")
                x, y = int(centerX - (width / 2)), int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_IDs.append(classID)
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, confidence_threshold, overlap_threshold
    )

    # Map from YOLO labels to Udacity labels.
    UDACITY_LABELS = {
        0: "pedestrian",
        1: "biker",
        2: "car",
        3: "biker",
        5: "truck",
        7: "truck",
        9: "trafficLight",
    }
    xmin, xmax, ymin, ymax, labels, scores = [], [], [], [], [], []
    if len(indices) > 0:
        # loop over the indexes we are keeping
        for i in indices.flatten():
            label = UDACITY_LABELS.get(class_IDs[i], None)
            if label is None:
                continue

            # extract the bounding box coordinates
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]

            xmin.append(x)
            ymin.append(y)
            xmax.append(x + w)
            ymax.append(y + h)
            labels.append(label)
            scores.append(confidences[i])

    boxes = pd.DataFrame(
        {
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "label": labels,
            "confidence": scores,
        }
    )

    return boxes


def display_images_with_bbox(image, boxes, title):
    LABEL_COLORS = {
        "car": ["LightSkyBlue", "RoyalBlue"],
        "pedestrian": ["red", "darkred"],
        "truck": ["green", "darkgreen"],
        "trafficLight": ["lightyellow", "yellow"],
        "biker": ["orange", "darkorange"],
    }
    img_height, img_width = image.shape[:2]

    fig = go.Figure()
    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.
    fig.add_trace(
        go.Scatter(
            x=[img_width * 0.05, img_width * 0.95],
            y=[img_height * 0.95, img_height * 0.05],
            mode="markers",
            marker_opacity=0,
            hoverinfo="none",
        )
    )
    fig.add_layout_image(
        dict(
            source=array_to_b64(image),
            x=0,
            y=0,
            xref="x",
            yref="y",
            sizex=img_width,
            sizey=img_height,
            sizing="stretch",
            opacity=1,
            layer="below",
        )
    )

    for index, box in boxes.iterrows():
        fill_col, line_col = LABEL_COLORS[box.label]
        add_bbox(
            fig,
            x0=box.xmin,
            y0=box.ymin,
            x1=box.xmax,
            y1=box.ymax,
            opacity=0.5,
            color=fill_col,
            name=f"class={box.label}<br>confidence={box.confidence:.3f}",
        )

    # Adapt axes to the right width and height, lock aspect ratio
    fig.update_xaxes(
        showgrid=False, visible=False, constrain="domain", range=[0, img_width]
    )
    fig.update_yaxes(
        showgrid=False,
        visible=False,
        scaleanchor="x",
        scaleratio=1,
        range=[img_height, 0],
    )
    fig.update_layout(title=title, showlegend=False)

    return fig


# Load labels and generate a summary
metadata = pd.read_csv("labels.csv.gz")
metadata["confidence"] = 1.0
summary = create_summary(metadata)

# Download model, and load it from the files
download_file(
    "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
    "yolov3.cfg",
)
download_file(
    "https://images.plot.ly/udacity-self-driving-cars/yolov3.weights", "yolov3.weights"
)
net, output_layer_names = load_network("yolov3.cfg", "yolov3.weights")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
server = app.server  # expose server variable for Procfile

# Create a cache, and cache all the heavy functions
cache = Cache(
    app.server,
    config={"CACHE_TYPE": "filesystem", 'CACHE_DIR': './cache/'},
)

load_image = cache.memoize()(load_image)
yolo_v3 = cache.memoize()(yolo_v3)

frame_controls = [
    NamedGroup(
        dbc.Select(
            id="dropdown-object",
            value=summary.columns[2],
            options=[{"label": i, "value": i} for i in summary.columns],
        ),
        label="Frame should contain:",
    ),
    dbc.ButtonGroup(
        [
            dbc.Button("Prev Frame", id="button-previous-frame", color='dark', outline=True, n_clicks=0,),
            dbc.Button("Next Frame", id="button-next-frame", color='dark', n_clicks=0,),
        ],
        style={'margin-top': '30px'}
    ),
    dbc.Button("Random Frame", id="button-random-frame", color='primary', n_clicks=0, style={'margin-top': '30px'}),
]

model_controls = [
    NamedGroup(
        dcc.Slider(
            id="slider-confidence",
            min=0,
            max=1,
            marks={0: "0.00", 1: "1.00"},
            step=0.01,
            value=0.5,
            tooltip=dict(always_visible=True, placement="bottom"),
        ),
        label="Yolo v3 Confidence Threshold:",
    ),
    NamedGroup(
        dcc.Slider(
            id="slider-overlap",
            min=0,
            max=1,
            marks={0: "0.00", 1: "1.00"},
            step=0.01,
            value=0.5,
            tooltip=dict(always_visible=True, placement="bottom"),
        ),
        label="Yolo v3 Overlap Threshold:",
    ),
]


app.layout = dbc.Container(
    fluid=True,
    children=[
        html.H1("YOLO Real-Time Object Detection"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="graph-ground-truth", style={"height": "calc(85vh - 150px)", "min-height": "500px"})),
                dbc.Col(dcc.Graph(id="graph-yolo-v3", style={"height": "calc(85vh - 150px)", "min-height": "500px"})),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Row(
                        [dbc.Col(c) for c in frame_controls]
                    )
                ),
                dbc.Col(
                    dbc.Row(
                        [dbc.Col(c) for c in model_controls]
                    )
                ),
            ]
        ),
    ],
)


@app.callback(
    Output("button-next-frame", "n_clicks"),
    [Input("button-random-frame", "n_clicks")],
    [State("button-next-frame", "n_clicks")],
)
def select_random_frame(_, next_n_clicks):
    return next_n_clicks + 50


@app.callback(
    [Output("graph-ground-truth", "figure"), Output("graph-yolo-v3", "figure")],
    [
        Input("dropdown-object", "value"),
        Input("button-previous-frame", "n_clicks"),
        Input("button-next-frame", "n_clicks"),
        Input("slider-confidence", "value"),
        Input("slider-overlap", "value"),
    ],
)
def update_graphs(
    object_type, prev_n_clicks, next_n_clicks, confidence_threshold, overlap_threshold
):
    t1 = time.time()
    min_elts, max_elts = 1, 25

    selected_frames = get_selected_frames(summary, object_type, min_elts, max_elts)
    num_frames = len(selected_frames)
    selected_frame_index = (next_n_clicks - prev_n_clicks) % num_frames

    selected_frame = selected_frames[selected_frame_index]

    img_bucket = "https://images.plot.ly/udacity-self-driving-cars/"
    image_url = os.path.join(img_bucket, selected_frame)
    image = load_image(image_url)
    t2 = time.time()

    real_boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    real_fig = display_images_with_bbox(image, real_boxes, "Human Annotated Frame")

    t3 = time.time()
    yolo_boxes = yolo_v3(
        image, net, output_layer_names, confidence_threshold, overlap_threshold
    )
    yolo_fig = display_images_with_bbox(image, yolo_boxes, "Yolo v3 Annotated Frame")

    t4 = time.time()
    print(
        f"Finished predictions in {t4-t3:.2f}s. Loaded Image in {t2-t1:.2f}s. Total time: {t4-t1:.2f}s"
    )

    return real_fig, yolo_fig


if __name__ == "__main__":
    app.run_server(debug=True)
