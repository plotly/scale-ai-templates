# ========== (c) JP Hwang 9/5/20  ==========

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
import os

desired_width = 320
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", desired_width)

from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

flake_pw = os.getenv("SNOWSQL_PWD")
flake_warehouse = os.getenv("SNOWSQL_WAREHOUSE")

engine = create_engine(
    URL(
        account="dd20994.us-central1.gcp",
        user="nicolaskruchten",
        password=flake_pw,
        database="FOOD_REVIEWS",
        schema="public",
        warehouse=flake_warehouse,
        role="sysadmin",
    )
)

connection = engine.connect()

df = pd.read_sql_query("SELECT * FROM food_reviews where id < 100", engine)
