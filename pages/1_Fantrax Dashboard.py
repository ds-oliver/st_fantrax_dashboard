import os
import sys
import logging
import pandas as pd
import warnings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from constants import colors, divergent_colors
from files import new_matches_data, ros_data
from functions import load_css, add_construction, create_custom_cmap,create_custom_divergent_cmap, style_dataframe_custom, round_and_format

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
    page_icon=":soccer:",
    layout="wide"
)

load_css()

def load_csv_file(csv_file):
    return pd.read_csv(csv_file)

warnings.filterwarnings('ignore')

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
sys.path.append(scripts_path)

@st.cache_data
def load_cached_css():
    load_css()

def load_csv_file_cached(csv_file, set_index_cols=None):
    """
    Loads a CSV file and applies a function to round and format its values.
    Optionally sets one or more columns as the DataFrame index.

    Parameters:
        csv_file (str): Path to the CSV file.
        set_index_cols (list, str, optional): Column(s) to set as DataFrame index. Defaults to None.

    Returns:
        pd.DataFrame: The loaded and formatted DataFrame.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file).applymap(round_and_format)
    
    # Check if set_index_cols is provided
    if set_index_cols:
        # Check if all columns in set_index_cols exist in the DataFrame
        missing_cols = [col for col in set_index_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame. Cannot set as index.")
        
        # Set the specified columns as the DataFrame index
        df.set_index(set_index_cols, inplace=True)
        
    return df

@st.cache_data
def create_custom_cmap_cached(*colors):
    return create_custom_cmap(*colors)

@st.cache_data
def create_custom_divergent_cmap_cached(*divergent_colors):
    return create_custom_divergent_cmap(*divergent_colors)

# Cache this function to avoid re-styling the DataFrame every time
@st.cache_data
def display_dataframe(df, title, colors, divergent_colors, info_text=None):
    custom_cmap = create_custom_cmap(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap(*divergent_colors)
    columns_to_keep = df.columns.tolist()

    try:
        st.write(f"## {title}")
        logging.info(f"Attempting to style the {title} dataframe")
        styled_df = style_dataframe_custom(df, columns_to_keep, custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=600)
        logging.info(f"{title} Dataframe head: {df.head()}")
        logging.info(f"{title} Dataframe tail: {df.tail()}")
        if info_text:
            st.info(info_text)
    except Exception as e:
        logging.error(f"Error styling the {title} dataframe: {e}")
        st.error(f"Error styling the {title} dataframe: {e}")

def main():
    logging.info("Starting main function")
    add_construction()

    load_cached_css()

    logging.info("Creating custom color maps")
    custom_cmap = create_custom_cmap_cached(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap_cached(*divergent_colors)

    lastgw_df = load_csv_file_cached('data/display-data/recent_gw_data.csv')
    grouped_players_df = load_csv_file_cached('data/display-data/grouped_player_data.csv')
    team_df = load_csv_file_cached('data/display-data/team_data.csv', set_index_cols=['Team'])
    team_pos_df = load_csv_file_cached('data/display-data/team_pos_data.csv', set_index_cols=['Team', 'Position'])
    vs_team_df = load_csv_file_cached('data/display-data/vs_team_fbref.csv', set_index_cols=['team'])

    vs_team_pos_df = load_csv_file_cached('data/display-data/vs_team_pos_fbref.csv', set_index_cols=['team', 'position'])

    # get the most recent gameweek value
    last_gw = lastgw_df['GW'].max()

    # Use the cached function to display DataFrames
    display_dataframe(lastgw_df, f"GW **:orange[{last_gw}]** Player Data", colors, divergent_colors)
    st.info("Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW total. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value.")
    display_dataframe(grouped_players_df, "Player Data (All Gameweeks)", colors, divergent_colors, info_text=f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at **:orange[{max(lastgw_df['GW'])}]** gameweeks of data.")
    display_dataframe(team_df, "Team Data", colors, divergent_colors, info_text="Note: This table will show the statistics earned by each respective team, per game.")
    display_dataframe(team_pos_df, "Team, Position Data", colors, divergent_colors)
    display_dataframe(vs_team_df, "vsTeam Data (from FBRef)", colors, divergent_colors, info_text="Note: This table will show the statistics conceded by each respective team to the respective opponent, per game.")
    display_dataframe(vs_team_pos_df, "vsTeam by Position Data (from FBRef)", colors, divergent_colors, info_text="Note: This table will show the statistics conceded by each respective team to the respective opponent by position, per game.")
    # display_dataframe(vs_team_pos_df, "vsTeam Position Data", colors, divergent_colors)
  
    logging.info("Main function completed successfully")

if __name__ == "__main__":
    main()