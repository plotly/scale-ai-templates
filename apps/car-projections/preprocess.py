"""
Download vehicles.csv from: https://www.kaggle.com/austinreese/craigslist-carstrucks-data
Place it in the same directory as this script and run `python preprocess.py`.
"""
import pandas as pd

used = pd.read_csv('vehicles.csv')
used = used.drop(columns=['description', 'county', 'size'])
used = used.dropna()
used.to_csv('vehicles_preprocessed.csv', index=False)