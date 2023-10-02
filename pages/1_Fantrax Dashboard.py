import streamlit as st
import pandas as pd
import numpy as np
import sys
import logging
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
from concurrent.futures import ThreadPoolExecutor

from constants import stats_cols, shooting_cols, passing_cols, passing_types_cols, gca_cols, defense_cols, possession_cols, playing_time_cols, misc_cols, fbref_cats, fbref_leagues, matches_col_groups, matches_drop_cols, matches_default_cols, matches_drop_cols, matches_default_cols, matches_standard_cols, matches_passing_cols, matches_pass_types, matches_defense_cols, matches_possession_cols, matches_misc_cols, matches_default_cols_rename, matches_standard_cols_rename, matches_defense_cols_rename, matches_passing_cols_rename, matches_possession_cols_rename, matches_misc_cols_rename, matches_pass_types_rename, colors, divergent_colors, matches_rename_dict, colors, divergent_colors, matches_rename_dict

from files import fx_gw1_data as gw1_data, fx_gw2_data as gw2_data, fx_gw3_data

from functions import load_css, get_color, style_dataframe_custom, add_construction, debug_dataframe, create_custom_cmap, create_custom_sequential_cmap

# Set up relative path for the log file
current_directory = os.path.dirname(__file__)
log_file_path = os.path.join(current_directory, 'streamlit_app_logs.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'  # 'a' means append
)

st.set_page_config(
    page_title="Draft Alchemy",
    page_icon=":",
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


@st.cache_data
def load_only_csvs(directory_path):
    return [pd.read_csv(os.path.join(directory_path, filename)) for filename in os.listdir(directory_path) if filename.endswith('.csv')]


@st.cache_data
def load_and_concatenate_csvs(directory_path):
    logging.debug("Starting load_and_concatenate_csvs() function. Loading and concatenating CSVs.")

    def load_csv(file_path):
        df = pd.read_csv(file_path)
        df['GW'] = df['GP'].max()
        df_obj = df.select_dtypes(['object'])
        df[df_obj.columns] = df_obj.apply(lambda x: x.str.normalize(
            'NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
        return df

    with ThreadPoolExecutor() as executor:
        df_list = list(executor.map(load_csv, [os.path.join(
            directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith('.csv')]))

    logging.debug(f"Concatenating {len(df_list)} GWs worth of data into DataFrame.")

    concatenated_df = pd.concat(df_list, ignore_index=True)

    # Drop columns if they exist
    columns_to_drop = ['Pos', 'Status',
                       '+/-_ros', '+/-_gws', 'ADP', '%D', 'ID', 'Opponent']
    concatenated_df.drop(columns=[
                         col for col in concatenated_df.columns if col in columns_to_drop], axis=1, inplace=True)
    
    logging.debug(f"Unique GW values: {concatenated_df['GW'].unique()}")
    logging.debug("load_and_concatenate_csvs() function complete. Returning concatenated_df.")

    return concatenated_df

def drop_cols_if_exists(df, cols):
    df.drop(columns=[
        col for col in df.columns if col in cols], axis=1, inplace=True)

@st.cache_data
def merge_dfs(df1, df2):
    return pd.merge(df1, df2, on=['Player', 'Team'], how='inner', suffixes=('_ros', '_gws'))

def reorder_columns(df, cols):
    # order should be Player, Position, Team, GW,

def main():
    add_construction()

    custom_cmap = create_custom_sequential_cmap(*colors)
    custom_divergent_cmap = create_custom_sequential_cmap(*divergent_colors)

    fx_directory = 'data/fantrax-data'
    ros_directory = 'data/ros-data'

    # Load and debug data
    gws_df = load_and_concatenate_csvs(fx_directory)
    ros_df = load_only_csvs(ros_directory)[0]
    debug_dataframe(ros_df)
    debug_dataframe(gws_df)

    # Merge and style dataframes
    ros_gws_df = merge_dfs(ros_df, gws_df)

    columns_to_drop = ['Pos', 'Status',
                       '+/-_ros', '+/-_gws', 'ADP', '%D', 'ID', 'Opponent']

    # Drop columns if they exist
    drop_cols_if_exists(ros_gws_df, columns_to_drop)

    debug_dataframe(ros_gws_df)

    selected_columns = ros_gws_df.columns.tolist()
    styled_df = style_dataframe_custom(
        ros_gws_df, selected_columns, custom_cmap=custom_cmap, inverse_cmap=False, is_percentile=False)

    # Display the dataframe
    st.dataframe(
        ros_gws_df.style.apply(lambda _: styled_df, axis=None),
        use_container_width=True,
        height=(len(ros_gws_df))
    )


# init main function
if __name__ == "__main__":
    main()
