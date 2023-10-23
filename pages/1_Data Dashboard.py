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

def clear_cache_button():
    """
    Adds a button to the Streamlit sidebar to clear the cache.
    """
    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.experimental_rerun()

def main():

    data_path = 'data/display-data/final'

    logging.info("Starting main function")

    # Add a button to the sidebar to clear the cache
    clear_cache_button()

    add_construction()

    load_cached_css()

    # player data
    ## single gw: / recent_gw_players_df
    ## grouped: / grouped_players_df
    ## grouped: / set_piece_studs

    # team data
    ## single gw: / recent_gw_data_team
    ## grouped: / team_df
    ## grouped: / vs_team_df
    ## grouped: / home_team_byteam
    ## grouped: / away_team_byteam


    logging.info("Creating custom color maps")
    custom_cmap = create_custom_cmap_cached(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap_cached(*divergent_colors)

    recent_gw_players_df = load_csv_file_cached(f'{data_path}/recent_gw_data.csv')
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

    set_piece_studs = load_csv_file_cached(f'{data_path}/top_5_players_per_team.csv')
    set_piece_studs_teams = load_csv_file_cached(f'{data_path}/set_piece_stats_team.csv', set_index_cols=['team'])
    set_piece_studs_pos = load_csv_file_cached(f'{data_path}/set_piece_stats_pos.csv', set_index_cols=['ftx_position', 'position'])

    # get the most recent gameweek value
    recent_gw = all_gws_df['GW'].max()
    first_gw = all_gws_df['GW'].min()

    default_style = {
        "container": {"padding": "5!important", "background-color": "#08061c"},
        "icon": {"color": "#fefae0", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#6d597a"},
        "nav-link-selected": {"background-color": "#370617"}
    }
    
    # Create a dictionary to map dataframe names to actual dataframes and info_text
    df_dict = {
        # recent gw data
        "recent_gw_players_df": {
            "frames": [{
                # players who played in the most recent gameweek
                "title": f"Player Data (GW {recent_gw})",
                "data": recent_gw_players_df,
                "info_text": f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW total. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value."
            }, {
                # players who played in the most recent gameweek by team
                "title": f"GW {recent_gw} Team Data",
                "data": recent_gw_data_team,
                "info_text": f"Note: This table shows team-specific data for GW {recent_gw}."
            }, 
            ],
            "icon": "arrow-clockwise"
        },
        "grouped_players_df": {
            "frames": [{
                # all players, all gameweeks
                "title": f"Player Data (GW {first_gw} - {recent_gw})",
                "data": grouped_players_df,
                "info_text": f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data."
            }, 
            ],
            "icon": "minecart-loaded"
        },
        # set piece data
        "set_piece_studs": {
            "frames": [{
                "title": "Set Piece Studs",
                "data": set_piece_studs,
                "info_text": f"Note: This table shows the top 5 players per team for set piece statistics. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data."}, 
                {
                "title": "Set Piece Studs (Team)",
                "data": set_piece_studs_teams,
                "info_text": f"Note: This table shows the set piece statistics for each team. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data."}
        ],
            "icon": "fire"
        },
        "recent_gw_data_team": {
            "frames": [{
                "title": f"GW {recent_gw} Team Data",
                "data": recent_gw_data_team,
                "info_text": f"Note: This table shows team-specific data for GW {recent_gw}."
            }],
            "icon": "table"
        },
        "recent_gw_data_pos": {
            "frames": [{
                "title": f"GW {recent_gw} Position Data",
                "data": recent_gw_data_pos,
                "info_text": f"Note: This table shows position-specific data for GW {recent_gw}."
            }],
            "icon": "table"
        },
        "recent_gw_data_teampos": {
            "frames": [{
                "title": f"GW {recent_gw} Team, Position Data",
                "data": recent_gw_data_teampos,
                "info_text": f"Note: This table shows team-specific data by position for GW {recent_gw}."
            }],
            "icon": "table"
        }
    }

    # Modified dfs_to_display to correctly extract "title" and "icon"
    dfs_to_display = [(key, df_dict[key]["frames"][0]["title"], df_dict[key]["icon"]) for key in df_dict]
    dfs_keys = [df_key for df_key, _, _ in dfs_to_display]
    dfs_titles = [df_title for _, df_title, _ in dfs_to_display]
    dfs_icons = [df_icon for _, _, df_icon in dfs_to_display]

    # Use default_style for all dataframes
    with st.sidebar:
        selected_df_name = option_menu("Select DataFrame", dfs_keys,
                                    icons=dfs_icons, menu_icon="list",
                                    styles=default_style)

    # Find the key associated with the selected DataFrame title
    selected_df_key = [key for key, title, _ in dfs_to_display if title == selected_df_name][0]

    # Conditionally display the selected DataFrame and info text
    if selected_df_key:
        st.toast("Loading data...")

        selected_frames = df_dict.get(selected_df_key, {}).get('frames', [])
        for frame in selected_frames:
            display_dataframe(frame["data"], frame["title"], colors, divergent_colors, info_text=frame["info_text"])
    else:
        st.error(f"DataFrame '{selected_df_name}' not found where expected.")

    logging.info("Main function completed successfully")


if __name__ == "__main__":
    main()