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
import time
import numpy as np
import mplsoccer as mpl
from mplsoccer import Bumpy
import matplotlib.pyplot as plt
import matplotlib.colors

from constants import simple_colors, divergent_colors
from files import new_matches_data, ros_data
from functions import load_css, add_construction, create_custom_cmap,create_custom_divergent_cmap, style_dataframe_custom, round_and_format, add_datadump_info

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

# load_css()

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
def create_custom_cmap_cached(*simple_colors):
    return create_custom_cmap(*simple_colors)

@st.cache_data
def create_custom_divergent_cmap_cached(*divergent_colors):
    return create_custom_divergent_cmap(*divergent_colors)

# Cache this function to avoid re-styling the DataFrame every time
@st.cache_data
def display_dataframe(df, title, simple_colors, divergent_colors, info_text=None, upper_info_text=None, drop_cols=[]):
    df = df.copy()

    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    custom_cmap = create_custom_cmap(*simple_colors)
    custom_divergent_cmap = create_custom_divergent_cmap(*divergent_colors)
    columns_to_keep = df.columns.tolist()

    # Dynamically calculate the height based on the number of rows
    # Set a minimum height of 300 and a maximum height of 800
    height = max(400, min(800, df.shape[0] * 25))

    try:
        st.write(f"## {title}")

        if upper_info_text:
            st.info(upper_info_text)

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

@st.cache_data
def get_sell_high_players(df, head=50):
    # create copy
    df = df.copy()
    print("printing df columns")
    print(df.columns)
    # sort by overperformance
    if 'Overperformance' in df.columns and 'HeatStreak' in df.columns:
        # get subset of data where Overperformance is greater than 0
        df = df[df['Overperformance'] > 0]
        # convert Overperformance to int and HeatStreak to float
        df['Overperformance'] = df['Overperformance'].astype(int)
        df['HeatStreak'] = df['HeatStreak'].astype(float).apply(lambda x: round(x, 2))
        # OvpAgg should be Overperformance * HeatStreak
        df['OvpAgg'] = round((df['Overperformance'] * df['HeatStreak']), 2)
        # sort by OvpAgg
        df = df.sort_values(by=['OvpAgg'], ascending=False).head(head)
        # reset index
        df = df.reset_index(drop=True)
        # return top 50
        return df
    
def get_date_created(file_path: str) -> str:
    """
    Get the creation date of a file.

    Parameters:
        file_path (str): The path of the file.

    Returns:
        str: The creation date in human-readable format.
    """
    if os.path.exists(file_path):
        # Getting the timestamp for the file creation date
        timestamp = os.path.getctime(file_path)
        
        # Converting the timestamp to human-readable format
        date_created = time.strftime('%Y-%m-%d', time.localtime(timestamp))
        return date_created
    else:
        return "File does not exist."
    
def display_date_of_update(date_of_update, title="Last Data Refresh"):
    return st.write(f"{title}: {date_of_update}")

def plot_bumpy_chart(df, x_column, y_column, label_column, highlight_dict=None, **kwargs):
    if not all(col in df.columns for col in [x_column, y_column, label_column]):
        raise ValueError("The specified columns do not exist in the DataFrame.")

    x_list = sorted(df[x_column].unique())
    y_list = df[label_column].unique().tolist()

    # Create a dictionary of values for plotting
    values = {}
    for player in y_list:
        player_df = df[df[label_column] == player]
        values[player] = player_df[y_column].tolist()

    # Create the Bumpy object and plot the data
    bumpy = Bumpy(**kwargs)
    bumpy.plot(x_list, y_list, values, highlight_dict=highlight_dict)


def main():

    data_path = 'data/display-data/final'

    logging.info("Starting main function")

    # Load the custom CSS file
    load_cached_css()

    # Add a button to the sidebar to clear the cache
    clear_cache_button()

    add_construction()

    file_path = "data/ros-data/Weekly ROS Ranks_GW11.csv"

    date_created = get_date_created(file_path)

    display_date_of_update(date_created, title="Ros Data Last Updated")

    # add_datadump_info()

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
    custom_cmap = create_custom_cmap_cached(*simple_colors)
    custom_divergent_cmap = create_custom_divergent_cmap_cached(*divergent_colors)

    recent_gw_players_df = load_csv_file_cached(f'{data_path}/recent_gw_data.csv')
    grouped_players_df = load_csv_file_cached(f'{data_path}/grouped_player_data.csv')
    all_gws_df = load_csv_file_cached(f'{data_path}/all_gws_data.csv')
    
    team_df = load_csv_file_cached(f'{data_path}/for_team.csv', set_index_cols=['team'])
    team_pos_df = load_csv_file_cached(f'{data_path}/d_detail_bypos_forteam.csv', set_index_cols=['team', 'position'])
    vs_team_df = load_csv_file_cached(f'{data_path}/vs_team.csv', set_index_cols=['opponent'])

    # vs_team_pos_df = load_csv_file_cached(f'{data_path}/vs_team_pos_fbref.csv', set_index_cols=['opponent', 'position'])

    all_pos = load_csv_file_cached(f'{data_path}/all_pos.csv', set_index_cols=['position'])
    ftx_pos_df = load_csv_file_cached(f'{data_path}/ftx_pos.csv', set_index_cols=['ftx_position'])

    # d_df_pos = load_csv_file_cached(f'{data_path}/d_detail_bypos.csv', set_index_cols=['position'])
    # m_df_pos = load_csv_file_cached(f'{data_path}/m_detail_bypos.csv', set_index_cols=['position'])
    # f_df_pos = load_csv_file_cached(f'{data_path}/f_detail_bypos.csv', set_index_cols=['position'])

    home_team_byteam = load_csv_file_cached(f'{data_path}/home_team_byteam.csv', set_index_cols=['team'])
    away_team_byteam = load_csv_file_cached(f'{data_path}/away_team_byteam.csv', set_index_cols=['team'])

    # # load all big_six_df_teampos, newly_promoted_df_teampos, rest_teams_df_teampos, spotlight_teams_df_teampos
    big_six_teampos = load_csv_file_cached(f'{data_path}/big_six_teampos.csv', set_index_cols=['team', 'position'])
    newly_promoted_teampos = load_csv_file_cached(f'{data_path}/newly_promoted_teampos.csv', set_index_cols=['team', 'position'])
    rest_teams_teampos = load_csv_file_cached(f'{data_path}/rest_teams_teampos.csv', set_index_cols=['team', 'position'])
    spotlight_teams_teampos = load_csv_file_cached(f'{data_path}/spotlight_teams_teampos.csv', set_index_cols=['team', 'position'])

    recent_gw_data_team = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_team.csv', set_index_cols=['Team'])
    recent_gw_data_pos = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_pos.csv', set_index_cols=['Position'])
    recent_gw_data_teampos = load_csv_file_cached(f'{data_path}/recent_gw_data_agg_team_pos.csv', set_index_cols=['Team', 'Position'])

    set_piece_studs = load_csv_file_cached(f'{data_path}/top_5_players_per_team.csv')
    set_piece_studs_teams = load_csv_file_cached(f'{data_path}/set_piece_stats_team.csv', set_index_cols=['team'])

    # call get_sell_high_players
    sell_high_players = get_sell_high_players(recent_gw_players_df, head=50)

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
        f"Recent GW Data (GW {recent_gw})": {
            "frames": [{
                # players who played in the most recent gameweek
                "title": f"Player Data (GW {recent_gw})",
                "data": recent_gw_players_df,
                "info_text": f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. GPR is a measure of Ghost Points over Total FPts; the higher the value the better a ghoster the player is. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW rolling sum of FPTS. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value."
            }, {
                # players who played in the most recent gameweek by team
                "title": f"GW {recent_gw} Team Data",
                "data": recent_gw_data_team,
                "info_text": f"Note: This table shows team-specific data for GW {recent_gw}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }, 
            ],
            "icon": "arrow-clockwise"
        },
        f"All GWs Data (GW {first_gw} - {recent_gw})": {
            "frames": [{
                # all players, all gameweeks
                "title": f"Player Data",
                "data": grouped_players_df,
                "info_text": f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                "upper_info_text": f"" 
            }, 
            {
                "title": f"Team Data",
                "data": team_df,
                "info_text": f"Note: This table shows team-specific data for all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"

            }, 
            {
                "title": f"Basic Positional Data",
                "data": ftx_pos_df,
                "info_text": f"Note: This table shows basic position-specific data for all gameweeks. These are the simple Fantrax positions including {', '.join(ftx_pos_df.index.get_level_values('ftx_position').unique().tolist())}. At this time we are looking at :orange[{recent_gw}] gameweeks of data.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            },
            {
                "title": f"Granular Positional Data",
                "data": all_pos,
                "info_text": f"Note: This table shows position-specific data for all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }
            ],
            "icon": "minecart-loaded"
        },
        # set piece data
        "Set Piece Data": {
            "frames": [{
                "title": "Set Piece Studs",
                "data": set_piece_studs,
                "info_text": f"Note: This table shows the top 5 players per team for set piece statistics. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data."
                }, 
                {
                "title": "Set Piece Studs (Team)",
                "data": set_piece_studs_teams,
                "info_text": f"Note: This table shows the set piece statistics for each team. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
                }
        ],
            "icon": "fire"
        },
        "Granular Team Data": {
            "frames": [{
                "title": f"Team Data (GW {first_gw} - {recent_gw})",
                "data": team_df,
                "info_text": f"Note: This table shows team-specific data for GW {recent_gw}.",
                "drop_cols": ["gp", "gp (max)", "gp (mean)"],
                "upper_info_text": f"Aggregated data is filtered first for players who played more than 45 minutes"  # Example columns to drop
            },{
                "title": f"vsTeam Data (GW {first_gw} - {recent_gw})",
                "data": vs_team_df,
                "info_text": f"Note: This table shows statistical categories teams have conceeded GW {recent_gw}.",
                "drop_cols": ["gp", "gp (max)", "gp (mean)"],  # Example columns to drop
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"

            }, {
                "title": f"Home Team Data (GW {first_gw} - {recent_gw})",
                "data": home_team_byteam,
                "info_text": f"Note: This table shows data team-specific data based on Home performances for GW {recent_gw}.",
                "drop_cols": ["gp", "gp (max)", "gp (mean)"],  # Example columns to drop
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"

            }, {
                "title": f"Away Team Data (GW {first_gw} - {recent_gw})",
                "data": away_team_byteam,
                "info_text": f"Note: This table shows data team-specific data based on Away performances for GW {recent_gw}.",
                "drop_cols": ["gp", "gp (max)", "gp (mean)"],  # Example columns to drop
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }
            ],
            "icon": "table"
        },
        "Teams Data by Status": {
            "frames": [{
                "title": f"Spotlight",
                "data": spotlight_teams_teampos,
                "info_text": f"Note: This table shows team-specific data by 'Spotlight' teams which include {', '.join(spotlight_teams_teampos.index.get_level_values('team').unique().tolist())}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }, {
                "title": f"Newly Promoted",
                "data": newly_promoted_teampos,
                "info_text": f"Note: This table shows team-specific data by 'Newly Promoted' teams which include {', '.join(newly_promoted_teampos.index.get_level_values('team').unique().tolist())}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }, {
                "title": f"Rest of League",
                "data": rest_teams_teampos,
                "info_text": f"Note: This table shows team-specific data by 'Rest of League' teams which include {', '.join(rest_teams_teampos.index.get_level_values('team').unique().tolist())}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }],
            "icon": "bar-chart-steps"
        },
        # positional data
        "Granular Positional Data": {
            "frames": [{
                "title": f"Positional Data (GW {first_gw} - {recent_gw})",
                "data": all_pos,
                "info_text": f"Note: This table shows position-specific data for GW {recent_gw}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }, {
                "title": f"Team Positional Data (GW {first_gw} - {recent_gw})",
                "data": team_pos_df,
                "info_text": f"Note: This table shows team-specific position data for GW {recent_gw}.",
                "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes"
            }
            ],
            "icon": "gem"
        },
        "Fantasy Insights": {
            "frames": [{
                "title": f"Sell-High Candidates",
                "data": sell_high_players,
                "info_text": f"Note: This table shows top 50 players who are currently overperforming their RoS value and are on a good run of form."
            }
            ],
            "icon": "lightbulb"
        },
        "Bumpy Charts": {
        "frames": [
            {
                "title": "Bumpy Chart Example",
                "type": "bumpy",
                "data": all_gws_df,  # Example DataFrame
                "x_column": "GW",
                "y_column": "Rank",
                "label_column": "Player",
                "highlight_dict": {"Player A": "red", "Player B": "blue"}  # Example highlight_dict

            }
        ],
        "icon": "chart-line"
        }
    }

    # List of the DataFrames to display based on the keys in the df_dict
    dfs_to_display = [(key, df_dict[key]["icon"]) for key in df_dict]

    # Define icons for each DataFrame
    dfs_keys = [df_key for df_key, _ in dfs_to_display]
    dfs_icons = [df_icon for _, df_icon in dfs_to_display]

    # Streamlit Option Menu for DataFrame selection
    with st.sidebar:
        selected_df_key = option_menu(
                    "Select DataFrame", dfs_keys,
                    icons=dfs_icons,
                    menu_icon="list",
                    styles=default_style)

    # Conditionally display the selected DataFrame and info text
    if selected_df_key:
        st.toast("Loading data...")
        selected_frames = df_dict.get(selected_df_key, {}).get('frames', [])
        for frame in selected_frames:
            if frame.get("type") == "bumpy":
                # Get the list of available numerical metrics (columns) and players from the DataFrame
                available_metrics = [col for col in frame['data'].select_dtypes(include=[np.number]).columns if col not in [frame['x_column'], frame['label_column']]]
                available_players = frame['data'][frame['label_column']].unique().tolist()

                # Streamlit widgets to let the user select the metric and the players
                selected_metric = st.selectbox('Select Metric for Y-axis', available_metrics)
                selected_players = st.multiselect('Select up to 3 Players to Highlight', available_players, default=available_players[:3])

                # Filter the DataFrame based on the selected metric and players
                filtered_df = frame['data'][frame['data'][frame['label_column']].isin(selected_players)]

                # Generate highlight_dict based on user selection
                colors = plt.cm.rainbow(np.linspace(0, 1, len(selected_players)))
                highlight_dict = {player: matplotlib.colors.to_hex(colors[i]) for i, player in enumerate(selected_players)}

                plot_bumpy_chart(filtered_df, frame['x_column'], selected_metric, frame['label_column'], highlight_dict=highlight_dict)


            else:
                display_dataframe(frame["data"], frame["title"], colors, divergent_colors, 
                                info_text=frame.get("info_text"), 
                                upper_info_text=frame.get("upper_info_text"),
                                drop_cols=frame.get("drop_cols", []))
    else:
        st.error(f"DataFrame '{selected_df_key}' not found where expected.")


    logging.info("Main function completed successfully")


if __name__ == "__main__":
    main()