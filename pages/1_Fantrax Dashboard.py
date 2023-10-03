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

from files import matches_data, ros_data

from functions import load_css, get_color, style_dataframe_custom, add_construction, debug_dataframe, create_custom_cmap, create_custom_sequential_cmap

# Set up relative path for the log file
current_directory = os.path.dirname(__file__)
log_file_path = os.path.join(current_directory, 'info_streamlit_app_logs.log')

print(f"Log file path: {log_file_path}")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
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

def main():
    logging.info("Starting main function")
    print("Starting main function")
    add_construction()

    logging.info("Creating custom color maps")
    custom_cmap = create_custom_sequential_cmap(*colors)
    custom_divergent_cmap = create_custom_sequential_cmap(*divergent_colors)
    logging.info("Custom color maps created")

    df = pd.read_csv('cleaned_transformed_data.csv')

    columns_to_keep = df.columns.tolist()

    try:
        logging.info("Attempting to style the final dataframe")
        logging.info(f"Columns in all_data before styling: {df.columns.tolist()}")
        
        styled_df = style_dataframe_custom(df, columns_to_keep, custom_cmap=custom_cmap, inverse_cmap=False, is_percentile=False)
        logging.info(f"Portion of styled DataFrame: {styled_df.head()}")
        
        logging.info("Styling completed, proceeding to display")
        st.dataframe(df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=1000)
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")

    logging.info("Main function completed successfully")

if __name__ == "__main__":
    main()
