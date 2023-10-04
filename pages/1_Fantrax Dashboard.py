import os
import sys
import logging
import pandas as pd
import warnings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from constants import colors, divergent_colors
from files import matches_data, ros_data
from functions import load_css, add_construction, create_custom_cmap,create_custom_divergent_cmap, style_dataframe_custom

# Set up relative path for the log file
current_directory = os.path.dirname(__file__)
log_file_path = os.path.join(current_directory, 'info_streamlit_app_logs.log')

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

st.set_page_config(
    page_title="Draft Alchemy",
    page_icon=":",
    layout="wide",  
    initial_sidebar_state="expanded"
)

load_css()

@st.cache_data
def load_csv_file(csv_file):
    return pd.read_csv(csv_file)

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

def main():
    logging.info("Starting main function")
    add_construction()

    logging.info("Creating custom color maps")
    custom_cmap = create_custom_cmap(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap(*divergent_colors)

    grouped_df = load_csv_file('all_game_weeks_data.csv')

    lastgw_df = load_csv_file('recent_gw_data.csv')

    columns_to_keep = lastgw_df.columns.tolist()

    try:
        st.write("## Most Recent Game Week Data")
        logging.info("Attempting to style the lastgw_df dataframe")
        styled_df = style_dataframe_custom(lastgw_df, columns_to_keep, custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(lastgw_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(lastgw_df) * 2)
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")

    logging.info("Main function completed successfully")

    # try:
    #     st.write("## ROS Data")
    #     logging.info("Attempting to style the grouped_df dataframe")
    #     styled_df = style_dataframe_custom(grouped_df, columns_to_keep, custom_cmap=custom_cmap, inverse_cmap=False, is_percentile=False)
    #     st.dataframe(grouped_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(grouped_df) * 20)
    # except Exception as e:
    #     st.write(f"An exception occurred: {e}")
    #     logging.error(f"An exception occurred: {e}")

    # logging.info("Main function completed successfully")

if __name__ == "__main__":
    main()
