"""
To generate the test files, please download [the dataset from Kaggle](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018?select=drugsComTrain_raw.csv), and extract `drugsComTest_raw.csv` in this folder.
"""
from datetime import datetime
import pandas as pd

test = pd.read_csv("drugsComTest_raw.csv")

test = test.dropna()
test = test[~test.condition.str.contains("found this comment")]

test.date = pd.to_datetime(test.date)
test["month"] = test.date + pd.offsets.MonthBegin(-1)

threshold = datetime(year=2016, month=1, day=1)

known = test[test.date < threshold].copy()
unknown = test[test.date > threshold].copy()

known["type"] = "known"
unknown["type"] = "predicted"

del unknown["rating"]

known.to_csv("old_reviews.csv", index=False)
unknown.to_csv("new_reviews.csv", index=False)
