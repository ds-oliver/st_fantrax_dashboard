import os
import sys
import logging
import pandas as pd
import warnings
import streamlit as st
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid
import streamlit_extras
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu

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

@st.cache_data
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
def display_dataframe(df, title, colors, divergent_colors, info_text=None):
    custom_cmap = create_custom_cmap(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap(*divergent_colors)
    columns_to_keep = df.columns.tolist()

    # Dynamically calculate the height based on the number of rows
    # Set a minimum height of 300 and a maximum height of 800
    height = max(400, min(800, df.shape[0] * 25))

    try:
        st.write(f"## {title}")
        logging.info(f"Attempting to style the {title} dataframe")
        styled_df = style_dataframe_custom(df, columns_to_keep, custom_cmap=custom_cmap, custom_divergent_cmap=custom_divergent_cmap, inverse_cmap=False, is_percentile=False)
        st.dataframe(df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True, height=height)
        logging.info(f"{title} Dataframe head: {df.head()}")
        logging.info(f"{title} Dataframe tail: {df.tail()}")
        if info_text:
            st.info(info_text)
    except Exception as e:
        logging.error(f"Error styling the {title} dataframe: {e}")
        st.error(f"Error styling the {title} dataframe: {e}")


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
    all_gws_df = load_csv_file_cached(f'{data_path}/all_gws_data.csv')
    
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

    recent_gw_data_team = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_team.csv', set_index_cols=['Team'])
    recent_gw_data_pos = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_pos.csv', set_index_cols=['Position'])
    recent_gw_data_teampos = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_team_pos.csv', set_index_cols=['Team', 'Position'])


    # get the most recent gameweek value
    last_gw = all_gws_df['GW'].max()
    first_gw = all_gws_df['GW'].min()
    
    # Create a dictionary to map dataframe names to actual dataframes and info_text
    df_dict = {
        "lastgw_df": {
            "title": f"GW {last_gw} Player Data",
            "data": lastgw_df,
            "info_text": f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW total. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value."
        },
        "recent_gw_data_team": {
            "title": f"GW {last_gw} Team Data",
            "data": recent_gw_data_team,
            "info_text": f"Note: This table shows team-specific data for GW {last_gw}."
        },
        "recent_gw_data_pos": {
            "title": f"GW {last_gw} Position Data",
            "data": recent_gw_data_pos,
            "info_text": f"Note: This table shows position-specific data for GW {last_gw}."
        },
        "recent_gw_data_teampos": {
            "title": f"GW {last_gw} Team, Position Data",
            "data": recent_gw_data_teampos,
            "info_text": f"Note: This table shows team-specific data by position for GW {last_gw}."
        },
        "grouped_players_df": {
            "title": f"Player Data GW {first_gw} - GW {last_gw}",
            "data": grouped_players_df,
            "info_text": f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at {max(lastgw_df['GW'])} gameweeks of data."
        },
        "team_df": {
            "title": "Team Data",
            "data": team_df,
            "info_text": "Note: This table shows team-specific data."
        },
        "team_pos_df": {
            "title": "Team, Position Data",
            "data": team_pos_df,
            "info_text": "Note: This table shows team-specific data by position."
        },
        "vs_team_df": {
            "title": "vsTeam Data (from FBRef)",
            "data": vs_team_df,
            "info_text": "Note: This table shows team-specific data against the respective opponent."
        },
        "all_pos": {
            "title": "All Positions Data",
            "data": all_pos,
            "info_text": "Note: This table shows data by position."
        },
        "d_df_pos": {
            "title": "Granular Defender Data",
            "data": d_df_pos,
            "info_text": "Note: This table shows data by defensive position."
        },
        "m_df_pos": {
            "title": "Granular Midfielder Data",
            "data": m_df_pos,
            "info_text": "Note: This table shows data by midfield position."
        },
        "f_df_pos": {
            "title": "Granular Forward Data",
            "data": f_df_pos,
            "info_text": "Note: This table shows data by forward position."
        },
        "home_team_byteam": {
            "title": "Home Team Data",
            "data": home_team_byteam,
            "info_text": "Note: This table shows data for the home team."
        },
        "away_team_byteam": {
            "title": "Away Team Data",
            "data": away_team_byteam,
            "info_text": "Note: This table shows data for the away team."
        },
        "spotlight_teams_teampos": {
            "title": "Spotlight Teams Data",
            "data": spotlight_teams_teampos,
            "info_text": f"Note: This data is comprised of specific positions ({spotlight_teams_teampos.index.get_level_values('position').unique().tolist()}) comprised of the following teams: {', '.join(spotlight_teams_teampos.index.get_level_values('team').unique().tolist())}."
        },
        "big_six_teampos": {
            "title": "Big Six Data",
            "data": big_six_teampos,
            "info_text": "Note: This table shows data for the big six teams by position."
        },
        "newly_promoted_teampos": {
            "title": "Newly Promoted Data",
            "data": newly_promoted_teampos,
            "info_text": "Note: This table shows data for the newly promoted teams by position."
        },
        "rest_teams_teampos": {
            "title": "Mid Table Data",
            "data": rest_teams_teampos,
            "info_text": "Note: This table shows data for the rest of the teams by position."
        },
        "big_six_teampos_d": {
            "title": "Big Six Data (Defenders)",
            "data": big_six_teampos_d,
            "info_text": "Note: This table shows data for the big six teams' defenders."
        }
    }

    # create a list of the DataFrames to display based on the titles in the df_dict
    dfs_to_display = [df_dict[key]["title"] for key in df_dict]

    # Streamlit Option Menu for DataFrame selection
    with st.sidebar:
        selected_df_name = option_menu("Select DataFrame", dfs_to_display, 
                                        icons=['table' for _ in range(len(dfs_to_display))],
                                        menu_icon="list")

    # Conditionally display the selected DataFrame and info text
    # Reverse lookup
    selected_df_key = [key for key, val in df_dict.items() if val["title"] == selected_df_name]

    if selected_df_key:
        st.toast("Loading data...")

        selected_df_key = selected_df_key[0]  # Take the first match
        selected_df_title = df_dict[selected_df_key]["title"]
        selected_df_data = df_dict[selected_df_key]["data"]
        selected_df_info_text = df_dict[selected_df_key]["info_text"]
        display_dataframe(selected_df_data, selected_df_title, colors, divergent_colors, info_text=selected_df_info_text)
    else:
        st.error(f"DataFrame '{selected_df_name}' not found where expected.")
        # display the first DataFrame in the list
        display_dataframe(df_dict[dfs_to_display[0]]["data"], df_dict[dfs_to_display[0]]["title"], colors, divergent_colors, info_text=df_dict[dfs_to_display[0]]["info_text"])


    logging.info("Main function completed successfully")


if __name__ == "__main__":
    main()