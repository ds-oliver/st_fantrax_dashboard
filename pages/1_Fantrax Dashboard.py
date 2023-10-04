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

    lastgw_df = load_csv_file('data/display-data/recent_gw_data.csv')

    # round to 2 decimal places
    for col in lastgw_df.columns:
        if lastgw_df[col].dtype == 'float64':
            lastgw_df[col] = lastgw_df[col].round(2)

    grouped_players_df = load_csv_file('data/display-data/grouped_player_data.csv')

    # round to 2 decimal places
    for col in grouped_players_df.columns:
        if grouped_players_df[col].dtype == 'float64':
            grouped_players_df[col] = grouped_players_df[col].round(2)

    team_df = load_csv_file('data/display-data/team_data.csv')

    # round to 2 decimal places
    for col in team_df.columns:
        if team_df[col].dtype == 'float64':
            team_df[col] = team_df[col].round(2)

    team_pos_df = load_csv_file('data/display-data/team_pos_data.csv')

    # round to 2 decimal places
    for col in team_pos_df.columns:
        if team_pos_df[col].dtype == 'float64':
            team_pos_df[col] = team_pos_df[col].round(2)

    columns_to_keep = lastgw_df.columns.tolist()

    try:
        st.write("## Most Recent Game Week Data")
        logging.info("Attempting to style the lastgw_df dataframe")
        styled_df = style_dataframe_custom(lastgw_df, columns_to_keep, custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(lastgw_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(lastgw_df) * 2)
        logging.info(f"Most Recent Game Week Dataframe head: {team_df.head()}")
        logging.info(f"Most Recent Game Week Dataframe tail: {team_df.tail()}")
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")

    columns_to_keep = grouped_players_df.columns.tolist()
    try:
        st.write("## Grouped Player Data")
        logging.info("Attempting to style the grouped_players_df dataframe")
        styled_df = style_dataframe_custom(grouped_players_df, grouped_players_df.columns.tolist(), custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(grouped_players_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(grouped_players_df) * 2)
        logging.info(f"Grouped Player Dataframe head: {grouped_players_df.head()}")
        logging.info(f"Grouped Player Dataframe tail: {grouped_players_df.tail()}")
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")

    columns_to_keep = team_df.columns.tolist()

    try:
        st.write("## Team Data")
        logging.info("Attempting to style the team_df dataframe")
        styled_df = style_dataframe_custom(team_df, team_df.columns.tolist(), custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(team_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(team_df) * 30)
        logging.info(f"Team Dataframe head: {team_df.head()}")
        logging.info(f"Team Dataframe tail: {team_df.tail()}")
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")


    columns_to_keep = team_pos_df.columns.tolist()
    
    try:
        st.write("## Team Position Data")
        logging.info("Attempting to style the team_pos_df dataframe")
        styled_df = style_dataframe_custom(team_pos_df, team_pos_df.columns.tolist(), custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(team_pos_df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=len(team_pos_df) * 10)
        logging.info(f"Team Position Dataframe head: {team_pos_df.head()}")
        logging.info(f"Team Position Dataframe tail: {team_pos_df.tail()}")
    except Exception as e:
        st.write(f"An exception occurred: {e}")
        logging.error(f"An exception occurred: {e}")

    logging.info("Main function completed successfully")

if __name__ == "__main__":
    main()
