# ========== (c) JP Hwang 28/4/20  ==========

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
import spacy
import json
from collections import defaultdict

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)

# Load data
fpath = "srcdata/amazon_fine_food_reviews.csv"
proc_fpath = "out/amazon_fine_food_reviews_proc.csv"
ner_df_fpath = "out/food_ner_df.csv"
df = pd.read_csv(fpath)

# # Use subset for testing
# df = df[:100000]
# proc_fpath = 'out/amazon_fine_food_reviews_proc_100k.csv'
# ner_df_fpath = 'out/food_ner_df_100k.csv'

textcols = ["Text", "Summary"]

for col in textcols:
    df[col] = df[col].fillna("")
    df[col] = df[col].str.replace(r"<.+>", "")
df = df.assign(ReviewDate=pd.to_datetime(df["Time"], unit="s"))

# Initialise spaCy model
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load('en_core_web_lg')

# Process & Collect NER tokens
ner_counts = defaultdict(int)
logger.info(f"Starting NLP analysis...")

for i, row in df.iterrows():
    ent_txts = list()
    for col in textcols:
        ner_col = "ner_str_" + col
        try:
            txt = row[col]
            doc = nlp(txt)

            ent_txts += [json.dumps([i.text, i.label_]) for i in doc.ents]
            df.loc[row.name, ner_col] = ";".join(ent_txts)
        except:
            logger.exception(f"Couldn't handle row {i}, col: {col}, txt: {row[col]}")
            df.loc[row.name, ner_col] = "ERROR"

    for ent in list(set(ent_txts)):
        ner_counts[ent] += 1

    if (i + 1) % 2000 == 0:
        logger.info(f"Processed {i+1} files out of {len(df)}.")
logger.info(f"Finished NLP analysis.")

# Build NER DB
ner_list = []
for k, v in ner_counts.items():
    ner = json.loads(k)
    ner_list.append(dict(ner=ner[0], label=ner[1], count=v))
ner_df = (
    pd.DataFrame(ner_list).sort_values("count", ascending=False).reset_index(drop=True)
)

# Save files
df.to_csv(proc_fpath, index=False)
ner_df.to_csv(ner_df_fpath, index=False)
