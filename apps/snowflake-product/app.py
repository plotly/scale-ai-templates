import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)

import pandas as pd
import numpy as np
import re
import json
import dash
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash.dependencies import Input, Output, State, ALL
import os
import plotly.express as px
import spacy
import textwrap
from textblob import TextBlob
import time
import ast
from collections import Counter

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine

def DataCard(width, value, label, id, color="primary"):
    return dbc.Card(
        body=True,
        color=color,
        style={'width': f"{width}%", 'margin-bottom': '10px'},
        children=[
            html.H3(id=id, children=value, className='card-title'),
            html.P(label, className='card-text')
        ],
        outline=True
    )


def NamedGroup(children, label, **kwargs):
    return dbc.FormGroup(
        [
            dbc.Label(label),
            *children
        ],
        **kwargs
    )


# ========== SET UP A CONNECTION TO SNOWFLAKE DB ==========
flake_pw = os.getenv("FLAKE_PW")
flake_user = os.getenv("FLAKE_USER")
flake_acc = os.getenv("FLAKE_ACCOUNT")

flake_db = "FOOD_REVIEWS"
flake_warehouse = "dash_snowflake"

query_cols = "id, helpful_num, helpful_denom, score, summary, text, review_date, ner_str_text, ner_str_summary"
def_query_lim = 1000

engine = create_engine(
    URL(
        account=flake_acc,
        user=flake_user,
        password=flake_pw,
        database=flake_db,
        schema="public",
        warehouse=flake_warehouse,
        role="sysadmin",
    ),
    pool_size=5,
    pool_recycle=1800,
    pool_pre_ping=True,
)

connection = engine.connect()

# ========================================
# Initialise SPACY
nlp = spacy.load("en_core_web_sm")

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)

# ========== SET DEFAULT PARAMETERS ==========
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.JOURNAL])
server = app.server

# Get constants to be used from Snowflake
db_size = connection.execute("SELECT count (*) FROM food_reviews").scalar()
mindate = connection.execute("SELECT MIN(review_date) FROM food_reviews").scalar()
maxdate = connection.execute("SELECT MAX(review_date) FROM food_reviews").scalar()

# ========== FUNCTIONS FOR DASH ==========


def unpack_filter_params(filter_params_json):
    # Unpack parameters from hidden layer
    logger.info(f"Parameters: {filter_params_json}")
    filter_params = json.loads(filter_params_json)

    filter_txt = filter_params[0]
    filter_yrs = filter_params[1]
    filter_ratings = filter_params[2]
    query_lim = filter_params[3]
    return filter_txt, filter_yrs, filter_ratings, query_lim


def get_filt_df(filter_params_json):
    # Get data from Snowflake

    filter_txt, filter_yrs, filter_ratings, query_lim = unpack_filter_params(
        filter_params_json
    )

    starttime = time.process_time()  # Get time taken for the query

    query_str = f"SELECT {query_cols} FROM {flake_db} SAMPLE ({query_lim} ROWS) WHERE "
    if filter_txt is not None:
        query_str += f"(LOWER(text) LIKE LOWER('%{filter_txt}%')) "
    query_str += (
        f"AND (score >= {filter_ratings[0]}) AND (score <= {filter_ratings[1]}) "
    )
    query_str += f"AND (review_date >= '{filter_yrs[0]}-01-01'::date) "
    query_str += f"AND (review_date <= '{filter_yrs[1]}-12-31'::date) "

    counter = 0
    got_df = False
    while (
        got_df == False and counter < 5
    ):  # Building in redundancies in case of SQL disconnect
        try:
            filt_df = pd.read_sql_query(query_str, engine)
            got_df = True
        except:
            counter += 1
            logger.error(
                f"Something went wrong there, trying again for attempt {1+counter}"
            )

    if got_df == False:
        filt_df = pd.DataFrame()

    tdelta = time.process_time() - starttime  # Time taken

    filt_df.fillna("", inplace=True)
    logger.info(f"Fetched {len(filt_df)} from SQL")

    return filt_df, tdelta


def build_rating_histogram(in_df):
    fig = px.histogram(
        in_df,
        x="score",
        template="plotly_white",
        color="Dataset",
        barmode="group",
        height=200,
        color_discrete_sequence=def_colors,
        labels={"count": "Count", "score": "Review Score"},
        histnorm="probability density",
    )
    fig.update_layout(margin={"t": 20}, legend={"y": 1.2})
    return fig


def build_date_histogram(in_df):
    fig = px.histogram(
        in_df,
        x="review_date",
        template="plotly_white",
        color="Dataset",
        barmode="group",
        nbins=40,
        height=200,
        color_discrete_sequence=def_colors,
        labels={"count": "Count", "review_date": "Review Date"},
        histnorm="probability density",
    )
    fig.update_layout(margin={"t": 20}, legend={"y": 1.2})
    return fig


# ========== Load data ==========
# # For local data (for testing)
# df = pd.read_csv('out/amazon_fine_food_reviews_proc_30k.csv', index_col=0)

# Get default SQL data - a random sampling
df, _ = get_filt_df(
    json.dumps(["", [mindate.year, maxdate.year], [1, 5], def_query_lim])
)

# Load precompiled NER data
ner_df = pd.read_csv("out/food_ner_df.csv")

used_ent_types = ["PERSON", "FAC", "ORG", "GPE", "PRODUCT"]
ner_filter = ner_df.label.isin(used_ent_types)

ner_df = ner_df.assign(ner_lower=ner_df["ner"].str.lower())
ner_df = ner_df.groupby("ner_lower")["count"].sum().sort_values()
ner_df = pd.DataFrame(ner_df).reset_index().rename({"ner_lower": "ner"}, axis=1)


def get_ents_list(txt):
    doc = nlp(txt)
    ents_list = list()
    for ent in doc.ents:
        if ent.label_ in used_ent_types:
            ents_list.append(ent.text)
    ents_list = list(set(ents_list))
    return ents_list


def get_n_hits(token, db_conn):
    token = token.replace("'", "\\'")

    counter = 0
    got_df = False
    while (
        got_df == False and counter < 5
    ):  # Building in redundancies in case of SQL disconnect
        try:
            n_hits = db_conn.execute(
                f"SELECT count (*) FROM food_reviews WHERE (LOWER(text) LIKE LOWER('%{token}%'))"
            ).scalar()
            got_df = True
        except:
            counter += 1
            logger.error(
                f"Something went wrong there, trying again for attempt {1+counter}"
            )
            n_hits = -1

    return n_hits


def fetch_rand_review(in_df, prev_id):
    quality_check = False
    # print(prev_id)
    counter = 0
    while quality_check == False:
        randind = np.random.randint(len(in_df))
        rand_review = in_df.iloc[randind]
        if (
            randind != prev_id
            and rand_review["text"] != ""
            and rand_review["summary"] != ""
        ):
            ents_list = get_ents_list(
                rand_review["summary"] + " " + rand_review["text"]
            )
            if len(ents_list) > 0:
                quality_check = True
        counter += 1
        if counter > 100 and quality_check == False:
            logger.warning(
                f"Could not find a row with useful data after 100 tries - check your code / data!"
            )
            break

    review = rand_review["summary"] + "\n\n" + rand_review["text"]
    return review, randind


def get_top_ners(in_df, n_hits=15):
    ner_counts = Counter()
    for i in in_df.ner_str_text.fillna("").values:
        for j in re.split(r"(?<=\]);(?=\[)", i):
            if len(j) > 0:
                try:
                    tmp_ner = ast.literal_eval(j)
                    if tmp_ner[1] in used_ent_types:
                        tmp_key = tmp_ner[0].lower()
                        ner_counts[tmp_key] += 1
                except:
                    logger.warning(f"Error parsing {i}")

    return ner_counts.most_common(n_hits)


def build_ner_freq_chart(top_ners, filt_db_size):

    ner_names = [i[0] for i in top_ners]

    ner_counts = [i[1] for i in top_ners]
    scaled_ner_counts = [int(i * db_size / filt_db_size) for i in ner_counts]
    def_ner_counts = [ner_df[ner_df.ner == i]["count"].values[0] for i in ner_names]

    all_counts = ner_counts + def_ner_counts
    log_all_counts = [np.log(i) for i in scaled_ner_counts] + [
        -np.log(i) for i in def_ner_counts
    ]

    tmp_df = pd.DataFrame(
        dict(
            NER=ner_names * 2,
            freq=log_all_counts,
            dataset=["Filtered"] * len(ner_names) + ["Overall"] * len(ner_names),
        )
    )
    tmp_df = tmp_df[::-1]

    fig = px.bar(
        tmp_df,
        title="Top Named Entities Found in Search Results",
        x="freq",
        y="NER",
        orientation="h",
        color="dataset",
        template="plotly_white",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        category_orders={"dataset": ["Overall", "Filtered"]},
        labels={"NER": "Named Entity", "freq": "Frequency"},
        hover_data="",
    )
    fig.update_layout(legend=dict(x=0.3, y=1.075), legend_orientation="h")
    fig.update_xaxes(showticklabels=False)
    fig.update_traces(hovertemplate=None, hoverinfo="skip")

    return fig


# ========== DASH LAYOUT ==========
def_colors = px.colors.carto.Safe


controls = [
    dbc.Card(
        [
            dbc.CardHeader(f"Overview:"),
            dbc.CardBody(
                [
                    dcc.Markdown(
                        textwrap.dedent(
                            f"""
            This app utilises Snowflake to efficiently search through {db_size} reviews for analysis.

            Use the filters below to navigate the dataset, or enter your own review to find similar products.
            """
                        )
                    ),
                ]
            )
        ]
    ),
    html.Br(),
    dbc.Card(
        [
            dbc.CardHeader("Search Snowflake Database"),
            dbc.CardBody(
                [
                    NamedGroup(
                        [
                            dbc.Input(
                                id="text-filter", type="text", value="Starbucks"
                            )
                        ],
                        label='Find reviews with words: (e.g. try "Starbucks", "Oatmeal" or "Chocolate)"',
                    ),
                    dbc.Button(
                        "Search Snowflake DB",
                        id="search-button",
                        n_clicks=0,
                        color='primary'
                    ),
                    NamedGroup(
                        [
                            dcc.RangeSlider(
                                id="date-selector",
                                min=mindate.year,
                                max=maxdate.year,
                                step=1,
                                value=[mindate.year, maxdate.year],
                                marks={
                                    mindate.year: str(mindate.year),
                                    maxdate.year: str(maxdate.year),
                                },
                            ),
                        ],
                        label="Filter by Review Year (min, max)",
                    ),
                    NamedGroup(
                        [
                            dcc.RangeSlider(
                                id="rating-selector",
                                min=1,
                                max=5,
                                step=1,
                                value=[1, 5],
                                marks={
                                    1: "1 Star",
                                    2: "2",
                                    3: "3",
                                    4: "4",
                                    5: "5 Stars",
                                },
                            ),
                        ],
                        label="Filter by Review Rating (min, max)",
                    ),
                    NamedGroup(
                        [
                            dcc.Slider(
                                id="max-results-selector",
                                min=2,
                                max=4,
                                step=1,
                                marks={i: "{}".format(10 ** i) for i in [2, 3, 4]},
                                value=np.log10(def_query_lim),
                            ),
                        ],
                        label="Max. results",
                    ),
                ]
            )
        ]
    ),
]


main_body = [
    dbc.Card(
        [
            dbc.CardHeader("Review modeling (Try entering your own)"),
            dbc.CardBody([
                dbc.Textarea(
                    id="review-txt-input",
                    value="Fetch a review, or type your own review here.",
                    style={"width": "100%", "height": 125},
                ),
                html.Div(-1, id="review-txt-id", style={"display": "none"}),
                html.Div("", id="review-txt", style={"display": "none"}),
                dbc.Button(
                    "Fetch another review",
                    id="fetch-filt-review",
                    n_clicks=0,
                    color='info',
                    style={"width": "40%", "margin-right": "5px"},
                ),
                html.Small(
                    " (Try typing your own review and click away from the textbox!)"
                ),
                html.Div(id="ents-list-json", style={"display": "none"}),
            ])
        ]
    ),
    html.Br(), 
    dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            "Review Sentiment\n(-1: very bad, 1: very good)"
                        ),
                        dbc.CardBody(dbc.Spinner(dcc.Graph(id="rand-filt-review-sentiment"))),
                    ]
                ),
                md=7
            ),
            dbc.Col(
                dbc.Card([], id="rand-filt-review-keywords"),
                md=5
            )
        ]
    ),
    html.Br(),
    html.H3("Filtered data", id="filt_title"),
    html.Div(
        dbc.Alert(
            "Uh-oh: There are no hits matching that query - please try another query!",
            color='warning'
        ),
        style={"display": "none"},
        id="warning-div",
    ),
    html.Div(dbc.Card(id="filt-params", body=True), style={"display": "none"}),
    html.Br(), 
    dbc.Row(
        [
            dbc.Col(
                dbc.Spinner(dbc.Card(dcc.Graph(id="filt-ner-count"), body=True)), 
                md=7
            ),
            dbc.Col(
                [
                    dbc.Spinner(
                        DataCard(
                            id="search_speed",
                            width=80,
                            value=f"- ms",
                            label=f"Data retrieval time",
                            color="primary",
                        )
                    ),
                    dbc.Spinner(
                        DataCard(
                            id="dataset_size",
                            width=80,
                            value=f"{db_size}",
                            label=f"Reviews searched",
                            color="primary",
                        )
                    ),
                    dbc.Spinner(
                        DataCard(
                            id="n_hits",
                            width=80,
                            value=f"N/A",
                            label=f"Results found",
                            color="primary",
                        )
                    ),
                    dbc.Spinner(
                        DataCard(
                            id="avg_rating",
                            width=80,
                            value=f"N/A",
                            label=f"Avg. Rating (overall: N/A)",
                            color="success",
                        )
                    ),
                ],
                md=5,
            ),
        ]
    ),

    html.Br(),
    dbc.CardDeck(
        [
            dbc.Card(
                [
                    dbc.CardHeader(f"Review Counts by Score"),
                    dbc.CardBody(dbc.Spinner(dcc.Graph(id="filt-ratings-hist"))),
                ]
            ),
            dbc.Card(
                [
                    dbc.CardHeader(f"Review Counts by Date"),
                    dbc.CardBody(dbc.Spinner(dcc.Graph(id="filt-reviews-by-month"))),
                ]
            ),
        ]
    )
    
]

app.layout = dbc.Container(
    fluid=True,
    children=[
        # Control / sidebar block
        html.H1("Product Reviews Modeling with Snowflake"),
        html.Hr(),
        html.Br(), 
        dbc.Row(
            [
                dbc.Col(controls, md=4, id='controls-block'),
                dbc.Col(main_body, md=8)
            ]
        )
    ]
)


@app.callback(
    [
        Output("warning-div", "style"),
        Output("filt-params", "children"),
        Output("filt_title", "children"),
        Output("filt-ner-count", "figure"),
        Output("search_speed", "children"),
        Output("n_hits", "children"),
        Output("avg_rating", "children"),
        Output("avg_rating", "color"),
        Output("filt-reviews-by-month", "figure"),
        Output("filt-ratings-hist", "figure"),
    ],
    [
        Input("search-button", "n_clicks"),
        Input("date-selector", "value"),
        Input("rating-selector", "value"),
        Input("max-results-selector", "value"),
    ],
    [
        State("text-filter", "value"),
        State("review-txt-id", "children"),
        State("filt-params", "children"),
    ],
)
def update_filter_params(
    n_clicks,
    yr_range,
    rating_range,
    log_max_hits,
    filter_txt,
    prev_rev_id,
    old_filter_params,
):

    filter_txt = re.sub(
        "[^A-Za-z0-9 '&-]+", "", filter_txt
    )  # Strip search string to exclude special characters
    filter_txt = filter_txt.replace("'", "\\'")
    max_hits = 10 ** log_max_hits

    if filter_txt == "":
        filter_txt = None

    filter_params = json.dumps([filter_txt, yr_range, rating_range, max_hits])

    if old_filter_params == filter_params:  # Are the filter parameters unchanged?
        return (dash.no_update,) * 10

    else:
        # print(f'Update to input parameters detected - updating filter div based on {filter_params}')
        filt_df, tdelta = get_filt_df(filter_params)

        if len(filt_df) == 0:
            # print('No update possible - no matching data.')
            return ({"display": "block"},) + (dash.no_update,) * 9

        else:
            # Update NER frequency chart
            top_ners = get_top_ners(filt_df)
            top_ners_fig = build_ner_freq_chart(top_ners, len(filt_df))

            # Get color value for avg review rating
            avg_rating_col = px.colors.sequential.Greens[
                max(0, int(((filt_df.score.mean() - 3) / 2) * 8))
            ]

            # Concatenate query results w random sample (df)
            concat_df = pd.concat(
                [filt_df.assign(Dataset="Filtered"), df.assign(Dataset="Overall")],
                axis=0,
            )

            # print(f'Returning filter parameters {filter_params}.')
            return (
                {"display": "none"},
                filter_params,
                f"Filtered data ({filter_txt})",
                top_ners_fig,
                f"{round(tdelta * 1000, 1)} ms",
                len(filt_df),
                round(filt_df.score.mean(), 1),
                avg_rating_col,  # Update filtered data summary
                build_date_histogram(concat_df),
                build_rating_histogram(concat_df),
            )


# Update review being displayed
@app.callback(
    [
        Output("review-txt", "children"),
        Output("review-txt-id", "children"),
        Output("review-txt-input", "value"),
    ],
    [
        Input("filt-params", "children"),
        Input("fetch-filt-review", "n_clicks"),
        Input("review-txt-input", "n_blur"),
    ],
    [
        State("review-txt-id", "children"),
        State("review-txt-input", "value"),
        State("review-txt", "children"),
    ],
)
def update_review_div(
    filter_params,
    n_clicks,
    review_input_n_blur,
    prev_id,
    review_input_txt,
    prev_review_txt,
):
    ctx = dash.callback_context

    if ctx.triggered:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    else:
        trigger_id = None

    if trigger_id != "review-txt-input":
        filt_df, _ = get_filt_df(filter_params)
        review, randind = fetch_rand_review(filt_df, prev_id)
        return review, randind, review
    else:
        review = review_input_txt
        if prev_review_txt != review_input_txt:
            randind = -1
            return review, randind, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update


# Update review analysis div
@app.callback(
    [
        Output("rand-filt-review-sentiment", "figure"),
        Output("rand-filt-review-keywords", "children"),
        Output("ents-list-json", "children"),
    ],
    [Input("review-txt", "children")],
)
def update_rev_analysis_div(review_txt):

    blob = TextBlob(review_txt)
    cardaccent_ch = int(np.clip((blob.sentiment.polarity + 0.5) // 0.1, 0, 10))
    # Compress range to -0.5 to 0.5 for higher visual impact
    sentiment_col = px.colors.diverging.RdBu[cardaccent_ch]
    sentiment_val = str(round(blob.sentiment.polarity, 2))

    fig = (
        px.bar(x=[sentiment_val], orientation="h", range_x=[-1, 1], height=100,)
        .update_yaxes(visible=False)
        .update_xaxes(title="")
        .update_traces(marker=dict(color=sentiment_col))
        .update_layout(margin=dict(b=0))
    )

    ents_list = get_ents_list(review_txt)

    # In some deployments - the engine seemed to be timing out (only for the n_hits queries)
    # Creating new ones here as a result.
    sm_engine = create_engine(
        URL(
            account="dd20994.us-central1.gcp",
            user="nicolaskruchten",
            password=flake_pw,
            database=flake_db,
            schema="public",
            warehouse=flake_warehouse,
            role="sysadmin",
        ),
        pool_size=2,
        pool_recycle=600,
        pool_pre_ping=True,
    )
    tmp_conn = sm_engine.connect()
    n_hits_list = [
        get_n_hits(token, tmp_conn) for token in ents_list
    ]  # Get list of matching hits on DB
    tmp_conn.close()  # Close connection when finished

    keywords = []
    for i in range(len(ents_list)):
        ent = ents_list[i]
        if n_hits_list[i] == -1:
            keywords.append(
                dbc.Button(
                    ent,
                    id={"type": "filter-review-ent", "index": i},
                    style={"margin-right": "5px"},
                    color='primary'
                )
            )
        elif n_hits_list[i] > 0:  # Do not provide links if no results are available
            keywords.append(
                dbc.Button(
                    ent + " (" + str(n_hits_list[i]) + " reviews)",
                    id={"type": "filter-review-ent", "index": i},
                    style={"margin-right": "5px"},
                    color='primary'
                )
            )
    if len(keywords) > 0:
        keywords = [dbc.CardHeader("See more reviews about:"), dbc.CardBody(keywords)]

    ents_list_json = json.dumps(ents_list)
    return fig, keywords, ents_list_json


# Callback to trigger a search from keyword click
@app.callback(
    [Output("text-filter", "value"), Output("search-button", "n_clicks")],
    [Input({"type": "filter-review-ent", "index": ALL}, "n_clicks")],
    [State("ents-list-json", "children"), State("search-button", "n_clicks"),],
)
def update_filt_from_rev_button(filt_review_btn, ents_list_json, old_n_clicks):
    trg = dash.callback_context.triggered
    # print(trg)
    if trg[0]["prop_id"] == ".":  # Upon app startup / no trigger
        # print('Initialising app, no update required')
        return dash.no_update, dash.no_update
    else:
        if trg[0]["value"] is None:  # Just a new review, no updated required
            return dash.no_update, dash.no_update
        else:  # Update filters
            btn_ind = json.loads(trg[0]["prop_id"].split(".")[0])["index"]
            ents_list = json.loads(ents_list_json)
            filt_str = ents_list[btn_ind]
            return filt_str, old_n_clicks + 1


if __name__ == "__main__":
    app.run_server(debug=False)
