import streamlit as st
import pandas as pd
import numpy as np
import sys
import unidecode
import os
# import logging
# import sqlite3
# import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
import warnings
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler
# import unicodedata
import plotly.graph_objects as go
# from bs4 import BeautifulSoup
import matplotlib.cm as mpl_cm
from pandas.io.formats.style import Styler
import cProfile
import pstats
import io
import matplotlib.colors as mcolors
import matplotlib
from collections import Counter
# from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
# from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.stylable_container import stylable_container
# from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import percentileofscore

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_col_groups, matches_drop_cols, matches_default_cols, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_default_cols_rename, matches_standard_cols_rename, matches_defense_cols_rename, matches_passing_cols_rename, matches_possession_cols_rename, matches_misc_cols_rename, matches_pass_types_rename, colors, divergent_colors, matches_rename_dict, colors, divergent_colors, matches_rename_dict

from files import fx_gw1_data as gw1_data, fx_gw2_data as gw2_data, fx_gw3_data

from functions import load_css, get_color, get_divergent_color, style_dataframe_custom, add_construction, debug_dataframe

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide",  
    initial_sidebar_state="expanded",
    menu_items={

    }
)

# Load the CSS file
# In pages/script.py
load_css()

# scraped data from : /Users/hogan/dev/fbref/scripts/rfx_scrape/fbref-scrape-current-year.py

# logger = st.logger

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)


def load_and_concatenate_csvs(directory_path):
    print("Debug: Starting load_and_concatenate_csvs() function. Loading and concatenating CSVs.")
    # Initialize an empty list to hold the dataframes
    df_list = []

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)

            # Load the CSV into a DataFrame and append it to the list
            df = pd.read_csv(file_path)

            # add GW column which is max of GP column
            df['GW'] = df['GP'].max()

            # apply unidecode to object columns
            df_obj = df.select_dtypes(['object'])
            df[df_obj.columns] = df_obj.apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

            df_list.append(df)

    print(f"Concatenating {len(df_list)} GWs worth of data into DataFrame.")

    # Concatenate all the dataframes in the list
    concatenated_df = pd.concat(df_list, ignore_index=True)

    # print unique GW values
    print(f"Unique GW values: {concatenated_df['GW'].unique()}")

    # print debug that function is complete
    print("Debug: load_and_concatenate_csvs() function complete. Returning concatenated_df.")

    return concatenated_df


def main():
    add_construction()

    directory_path = 'data/fantrax-data'
    final_df = load_and_concatenate_csvs(directory_path)

    debug_dataframe(final_df)

    styled_df = 


# init main function
if __name__ == "__main__":
    main()
