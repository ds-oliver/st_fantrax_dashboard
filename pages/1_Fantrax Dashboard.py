import os
import sys
import logging
import pandas as pd
import warnings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid

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
    df = pd.read_csv(csv_file, index_col=0 if 'Unnamed: 0' in pd.read_csv(csv_file).columns else None).applymap(round_and_format)
    
    # Check if set_index_cols is provided
    if set_index_cols:
        # Check if all columns in set_index_cols exist in the DataFrame
        missing_cols = [col for col in set_index_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame. Cannot set as index.\nDataFrame columns: {df.columns}")
        
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
def display_dataframe(df, title, colors, divergent_colors, info_text=None, use_expander=False, expander_label=None):
    custom_cmap = create_custom_cmap(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap(*divergent_colors)
    columns_to_keep = df.columns.tolist()

    # Dynamically calculate the height based on the number of rows
    # Set a minimum height of 300 and a maximum height of 800
    height = max(400, min(800, df.shape[0] * 25))

    # Choose either the main Streamlit instance or the expander to write to
    target_st = st
    if use_expander:
        expander_label = expander_label if expander_label else f"{title} (Click to expand)"
        target_st = st.expander(expander_label, expanded=False)

    try:
        target_st.write(f"## {title}")
        logging.info(f"Attempting to style the {title} dataframe")
        styled_df = style_dataframe_custom(df, columns_to_keep, custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        target_st.dataframe(df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=height)
        logging.info(f"{title} Dataframe head: {df.head()}")
        logging.info(f"{title} Dataframe tail: {df.tail()}")
        if info_text:
            target_st.info(info_text)
    except Exception as e:
        logging.error(f"Error styling the {title} dataframe: {e}")
        target_st.error(f"Error styling the {title} dataframe: {e}")

def set_index_based_on_radio_button(df, widget_key, df_name='DataFrame'):
    """
    Set DataFrame index based on a Streamlit radio button.

    Parameters:
        df (pd.DataFrame): DataFrame to modify.
        widget_key (str): Unique key for the radio widget.

    Returns:
        pd.DataFrame: DataFrame with "Player" set as index if radio button is ticked.
    """
    set_index_option = st.radio(f"Would you like to set 'Player' as the {df_name} table index?", ('No', 'Yes'), key=widget_key)
    if set_index_option == 'Yes':
        if 'Player' in df.columns:
            df.set_index('Player', inplace=True)
        else:
            st.warning("The DataFrame does not have a 'Player' column.")
    return df


def main():
    
    data_path = 'data/display-data/final'

    logging.info("Starting main function")
    add_construction()

    load_cached_css()

    logging.info("Creating custom color maps")
    custom_cmap = create_custom_cmap_cached(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap_cached(*divergent_colors)

    lastgw_df = load_csv_file_cached(f'{data_path}/recent_gw_data.csv')
    grouped_players_df = load_csv_file_cached(f'{data_path}/grouped_player_data.csv')
    
    team_df = load_csv_file_cached(f'{data_path}/for_team.csv', set_index_cols=['team'])
    team_pos_df = load_csv_file_cached(f'{data_path}/d_detail_bypos_forteam.csv', set_index_cols=['team', 'position'])
    vs_team_df = load_csv_file_cached(f'{data_path}/vs_team.csv', set_index_cols=['opponent'])

    # vs_team_pos_df = load_csv_file_cached(f'{data_path}/vs_team_pos_fbref.csv', set_index_cols=['opponent', 'position'])

    all_pos = load_csv_file_cached(f'{data_path}/all_pos.csv', set_index_cols=['position'])

    d_df_pos = load_csv_file_cached(f'{data_path}/d_detail_bypos.csv', set_index_cols=['position'])
    m_df_pos = load_csv_file_cached(f'{data_path}/m_detail_bypos.csv', set_index_cols=['position'])
    f_df_pos = load_csv_file_cached(f'{data_path}/f_detail_bypos.csv', set_index_cols=['position'])

    home_team_byteam = load_csv_file_cached(f'{data_path}/home_team_byteam.csv', set_index_cols=['team'])
    away_team_byteam = load_csv_file_cached(f'{data_path}/away_team_byteam.csv', set_index_cols=['team'])

    # load all big_six_df_teampos, newly_promoted_df_teampos, rest_teams_df_teampos, spotlight_teams_df_teampos
    big_six_teampos = load_csv_file_cached(f'{data_path}/big_six_teampos.csv', set_index_cols=['team', 'position'])
    newly_promoted_teampos = load_csv_file_cached(f'{data_path}/newly_promoted_teampos.csv', set_index_cols=['team', 'position'])
    rest_teams_teampos = load_csv_file_cached(f'{data_path}/rest_teams_teampos.csv', set_index_cols=['team', 'position'])
    spotlight_teams_teampos = load_csv_file_cached(f'{data_path}/spotlight_teams_teampos.csv', set_index_cols=['team', 'position'])

    # load all positional team group dfs
    big_six_teampos_d = load_csv_file_cached(f'{data_path}/big_six_players_d.csv', set_index_cols=['team', 'position'])
    big_six_teampos_m = load_csv_file_cached(f'{data_path}/big_six_players_m.csv', set_index_cols=['team', 'position'])
    big_six_teampos_f = load_csv_file_cached(f'{data_path}/big_six_players_f.csv', set_index_cols=['team', 'position'])
    newly_promoted_teampos_d = load_csv_file_cached(f'{data_path}/newly_promoted_players_d.csv', set_index_cols=['team', 'position'])
    newly_promoted_teampos_m = load_csv_file_cached(f'{data_path}/newly_promoted_players_m.csv', set_index_cols=['team', 'position'])
    newly_promoted_teampos_f = load_csv_file_cached(f'{data_path}/newly_promoted_players_f.csv', set_index_cols=['team', 'position'])
    rest_teams_teampos_d = load_csv_file_cached(f'{data_path}/rest_teams_players_d.csv', set_index_cols=['team', 'position'])
    rest_teams_teampos_m = load_csv_file_cached(f'{data_path}/rest_teams_players_m.csv', set_index_cols=['team', 'position'])
    rest_teams_teampos_f = load_csv_file_cached(f'{data_path}/rest_teams_players_f.csv', set_index_cols=['team', 'position'])
    spotlight_teams_teampos_d = load_csv_file_cached(f'{data_path}/spotlight_teams_players_d.csv', set_index_cols=['team', 'position'])
    spotlight_teams_teampos_m = load_csv_file_cached(f'{data_path}/spotlight_teams_players_m.csv', set_index_cols=['team', 'position'])
    spotlight_teams_teampos_f = load_csv_file_cached(f'{data_path}/spotlight_teams_players_f.csv', set_index_cols=['team', 'position'])


    # get the most recent gameweek value
    last_gw = lastgw_df['GW'].max()

    lastgw_df = set_index_based_on_radio_button(lastgw_df, 'lastgw_df', df_name=f'GW {last_gw}')
    # Use the cached function to display DataFrames
    display_dataframe(lastgw_df, f"Player Data **(:orange[GW {last_gw}])**", colors, divergent_colors, info_text=f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW total. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value.")

    # Call the function to set the index based on radio button selection
    grouped_players_df = set_index_based_on_radio_button(grouped_players_df, 'grouped_players_df', df_name='All GWs')

    display_dataframe(grouped_players_df, "Player Data (All Gameweeks)", colors, divergent_colors, info_text=f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at **:orange[{max(lastgw_df['GW'])}]** gameweeks of data.")

    # display spotlight_teams_teampos_f
    display_dataframe(spotlight_teams_teampos_f, "Spotlight Teams' Forwards' Data", colors, divergent_colors, info_text=f"Note: This data is comprised of specific positions ({spotlight_teams_teampos_f.index.get_level_values('position').unique().tolist()}) comprised of the following teams: {', '.join(spotlight_teams_teampos_f.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Spotlight Teams' Forwards' Data (Click to expand)") 
    display_dataframe(spotlight_teams_teampos_m, "Spotlight Teams' Midfielders' Data", colors, divergent_colors, info_text=f"Note: This data is comprised of specific positions ({spotlight_teams_teampos_m.index.get_level_values('position').unique().tolist()}) comprised of the following teams: {', '.join(spotlight_teams_teampos_m.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Spotlight Teams' Midfielders' Data (Click to expand)")
    display_dataframe(spotlight_teams_teampos_d, "Spotlight Teams' Defenders' Data", colors, divergent_colors, info_text=f"Note: This data is comprised of specific positions ({spotlight_teams_teampos_d.index.get_level_values('position').unique().tolist()}) comprised of the following teams: {', '.join(spotlight_teams_teampos_d.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Spotlight Teams' Defenders' Data (Click to expand)")
    display_dataframe(big_six_teampos, "Big Six Data", colors, divergent_colors, info_text=f"Note: This data is comprised of the following teams: {', '.join(big_six_teampos.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Big Six Data (Click to expand)")
    display_dataframe(newly_promoted_teampos, "Newly Promoted Data", colors, divergent_colors, info_text=f"Note: This data is comprised of the following teams: {', '.join(newly_promoted_teampos.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Newly Promoted Data (Click to expand)")
    display_dataframe(rest_teams_teampos, "Mid Table Data", colors, divergent_colors, info_text=f"Note: This data is comprised of the following teams: {', '.join(rest_teams_teampos.index.get_level_values('team').unique().tolist())}.", use_expander=True, expander_label="Mid Table Data (Click to expand)")

    # display all_pos
    display_dataframe(all_pos, "All Positions Data", colors, divergent_colors, info_text="Note: This table will show the statistics by specific position, per game.", use_expander=True, expander_label="All Positions Data (Click to expand)")
    # display d_df_pos
    # display_dataframe(d_df_pos, "Granular Defender Data", colors, divergent_colors, info_text="Note: This table will show the statistics by specific defensive position, per game.")
    # display_dataframe(m_df_pos, "Granular Midfielder Data", colors, divergent_colors, info_text="Note: This table will show the statistics by specific midfield position, per game.")
    # display_dataframe(f_df_pos, "Granular Forward Data", colors, divergent_colors, info_text="Note: This table will show the statistics by specific forward position, per game.")

    # display home_team_byteam
    display_dataframe(home_team_byteam, "Home Team Data", colors, divergent_colors, info_text="Note: This table will show the statistics earned by each respective team in games played at home.", use_expander=True, expander_label="Home Team Data (Click to expand)")
    display_dataframe(away_team_byteam, "Away Team Data", colors, divergent_colors, info_text="Note: This table will show the statistics earned by each respective team in games played away from home.", use_expander=True, expander_label="Away Team Data (Click to expand)")
    display_dataframe(team_df, "Team Data", colors, divergent_colors, info_text="Note: This table will show the statistics earned by each respective team, per game.")
    display_dataframe(team_pos_df, "Team, Position Data", colors, divergent_colors, info_text="Note: This table will show the statistics earned by each respective team, by position, per game.")
    display_dataframe(vs_team_df, "vsTeam Data (from FBRef)", colors, divergent_colors, info_text="Note: This table will show the statistics conceded by each respective team to the respective opponent, per game.", use_expander=True, expander_label="vsTeam Data (from FBRef) (Click to expand)"
    # display_dataframe(vs_team_pos_df, "vsTeam by Position Data (from FBRef)", colors, divergent_colors, info_text="Note: This table will show the statistics conceded by each respective team to the respective opponent by position, per game.")
  
    logging.info("Main function completed successfully")


if __name__ == "__main__":
    main()