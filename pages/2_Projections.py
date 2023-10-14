import streamlit as st
import pandas as pd
import os
import sys
from warnings import filterwarnings
import base64
import streamlit_extras
from streamlit_extras.dataframe_explorer import dataframe_explorer
from markdownlit import mdlit
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.stylable_container import stylable_container
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.customize_running import center_running
import logging
from itertools import product

from constants import colors, divergent_colors
from files import projections as proj_csv, fx_gif, ros_ranks
from functions import load_csv, add_construction, load_css, create_custom_sequential_cmap, create_custom_cmap, create_custom_divergent_cmap, style_dataframe_custom, round_and_format, style_position_player_only

# set up logging
log_formatter = logging.Formatter("%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)  # Create logger with the script's name

# Set logging level
logger.setLevel(logging.INFO)

# Create a file handler and add it to the logger
current_directory = os.path.dirname(__file__)
log_file_path = os.path.join(current_directory, 'proj_logs.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

st.set_page_config(
    page_title="Footy Magic",
    page_icon=":soccer:",
    layout="wide"
)

load_css()

filterwarnings('ignore')

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
def display_dataframe_pos(df, title=None, info_text=None):
    columns_to_keep = df.columns.tolist()
    # apply round_and_format
    df = df.applymap(round_and_format)
    
    try:
        if title:
            st.write(f"### {title}")
        logging.info(f"Attempting to style the {title} dataframe")
        # Style the DataFrame
        styled_df = style_position_player_only(df, columns_to_keep)
        st.dataframe(df[columns_to_keep].style.apply(lambda _: styled_df, axis=None), use_container_width=True)
        logging.info(f"{title} Dataframe head: {df.head()}")
        logging.info(f"{title} Dataframe tail: {df.tail()}")
        if info_text:
            st.info(info_text)
    except Exception as e:
        logging.error(f"Error styling the {title} dataframe: {e}")
        st.error(f"Error styling the {title} dataframe: {e}")
        styled_df = df

def local_gif(file_path):
    with open(file_path, "rb") as file_:
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")

    return st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="download data" width="100%">',
        unsafe_allow_html=True,
    )

def debug_filtering(projections, players):
    # Ensure that the data frames are not empty
    if projections.empty or players.empty:
        logging.info("Debug - One or both DataFrames are empty.")
        return
    
    logging.info("Debug - Projections before filtering:", projections.head())
    logging.info("Debug - Players before filtering:", players.head())

    # Filter the projections DataFrame
    projections_filtered = projections[projections['ProjFPts'] >= 10]
    
    # Debug: Show filtered projections
    logging.info("Debug - Projections after filtering:", projections_filtered.head())

    # Filter the players DataFrame to keep only those in the filtered projections
    available_players = players[players['Player'].isin(projections_filtered['Player'])]
    
    # Debug: Show filtered players
    logging.info("Debug - available_players after filtering:", available_players.head())
    
    waivers_fa = ['Waivers', 'FA']

    # Filter the available players to remove players that are not in the "Waivers" or "FA" status
    filtered_available_players = players[players['Status'].isin(waivers_fa)]
    
    # Debug: Show filtered available_players
    logging.info("Debug - available_players:", filtered_available_players.head())

# def filter_by_status_and_position(players, projections, status):
#     if isinstance(status, str):
#         status = [status]

#     filtered_players = players[players['Status'].isin(status)]

#     if filtered_players.empty:
#         return pd.DataFrame(), pd.DataFrame(), 0

#     player_list = filtered_players['Player'].unique().tolist()
#     projections = projections[projections['Player'].isin(player_list)]

#     if projections.empty:
#         return pd.DataFrame(), pd.DataFrame(), 0

#     # Prioritize players with ProjGS not equal to 0
#     projections['Priority'] = projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
#     projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

#     pos_limits = {'D': (3, 5), 'M': (2, 5), 'F': (1, 3)}
#     max_players = 10
#     best_combination = None
#     best_score = 0

#     for d in range(pos_limits['D'][0], pos_limits['D'][1] + 1):
#         for m in range(pos_limits['M'][0], pos_limits['M'][1] + 1):
#             for f in range(pos_limits['F'][0], pos_limits['F'][1] + 1):
#                 if d + m + f != max_players:
#                     continue

#                 defenders = projections[projections['Position'] == 'D'].nlargest(d, 'ProjFPts')
#                 midfielders = projections[projections['Position'] == 'M'].nlargest(m, 'ProjFPts')
#                 forwards = projections[projections['Position'] == 'F'].nlargest(f, 'ProjFPts')

#                 current_combination = pd.concat([defenders, midfielders, forwards])
#                 current_score = current_combination['ProjFPts'].sum()

#                 if current_score > best_score:
#                     best_combination = current_combination
#                     best_score = current_score

#     logging.info(f"Total Defenders: {len(best_combination[best_combination['Position'] == 'D'])}")
#     logging.info(f"Total Midfielders: {len(best_combination[best_combination['Position'] == 'M'])}")
#     logging.info(f"Total Forwards: {len(best_combination[best_combination['Position'] == 'F'])}")

#     # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
#     best_combination.sort_values(by=['Position', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Position' else x, ascending=[True, False], inplace=True)
#     best_combination.reset_index(drop=True, inplace=True)

#     reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)
#     best_score = round(best_score, 1)

#     # Drop the 'Priority' column before returning
#     best_combination.drop(columns=['Priority'], inplace=True, errors='ignore')
#     reserves.drop(columns=['Priority'], inplace=True, errors='ignore')

#     return best_combination, reserves, best_score, projections

def filter_by_status_and_position(players, projections, status):
    if isinstance(status, str):
        status = [status]

    filtered_players = players[players['Status'].isin(status)]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, pd.DataFrame()

    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]

    if projections.empty:
        return pd.DataFrame(), pd.DataFrame(), 0, 0, pd.DataFrame()

    projections['Priority'] = projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    pos_limits = {'D': (3, 5), 'M': (2, 5), 'F': (1, 3)}
    max_players = 10
    best_combination = None
    best_score = 0
    best_starting_score = 0  # New variable to track the best score with only starters

    position_combinations = product(
        range(pos_limits['D'][0], pos_limits['D'][1] + 1),
        range(pos_limits['M'][0], pos_limits['M'][1] + 1),
        range(pos_limits['F'][0], pos_limits['F'][1] + 1)
    )

    for d, m, f in position_combinations:
        if d + m + f != max_players:
            continue

        defenders = projections[projections['Position'] == 'D'].nlargest(d, 'ProjFPts')
        midfielders = projections[projections['Position'] == 'M'].nlargest(m, 'ProjFPts')
        forwards = projections[projections['Position'] == 'F'].nlargest(f, 'ProjFPts')

        current_combination = pd.concat([defenders, midfielders, forwards])
        current_score = current_combination['ProjFPts'].sum()

        if current_score > best_score:
            best_combination = current_combination
            best_score = current_score
            best_starting_score = best_combination[best_combination['ProjGS'] != 0]['ProjFPts'].sum()  # Update the starting score

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)

    # Replace players with ProjGS == 0, if possible
    for _, player_row in best_combination.iterrows():
        if player_row['ProjGS'] == 0:
            pos = player_row['Position']
            replacement_candidates = reserves[(reserves['Position'] == pos) & (reserves['ProjGS'] != 0)]

            if not replacement_candidates.empty:
                replacement = replacement_candidates.iloc[0]
                best_combination = best_combination[best_combination['Player'] != player_row['Player']]
                best_combination = pd.concat([best_combination, replacement.to_frame().T])
                
                # Add the removed player back into the reserves
                reserves = pd.concat([reserves, player_row.to_frame().T])
                
                # Remove the replacement player from the reserves
                reserves = reserves[reserves['Player'] != replacement['Player']]

    # Sort DataFrame by 'Position' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Position', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Position' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    best_score = round(best_combination['ProjFPts'].sum(), 1)
    best_starting_score = round(best_combination[best_combination['ProjGS'] != 0]['ProjFPts'].sum(), 1)  # Finalize the starting score

    best_combination.drop(columns=['Priority'], inplace=True, errors='ignore')
    reserves.drop(columns=['Priority'], inplace=True, errors='ignore')

    return best_combination, reserves, best_score, best_starting_score, projections


# Filter available players by their ProjGS and status
def filter_available_players_by_projgs(players, projections, status, projgs_value):
    if isinstance(status, str):
        status = [status]

    filtered_players = players[players['Status'].isin(status)]
    
    if projgs_value is not None:
        filtered_players = filtered_players[filtered_players['ProjGS'] == projgs_value]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame()

    player_list = filtered_players['Player'].unique().tolist()
    projections = projections[projections['Player'].isin(player_list)]

    if projections.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Prioritize players with ProjGS not equal to 0
    projections['Priority'] = projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    pos_limits = {'D': (3, 5), 'M': (2, 5), 'F': (1, 3)}
    max_players = 10
    best_combination = None
    best_score = 0

    for d in range(pos_limits['D'][0], pos_limits['D'][1] + 1):
        for m in range(pos_limits['M'][0], pos_limits['M'][1] + 1):
            for f in range(pos_limits['F'][0], pos_limits['F'][1] + 1):
                if d + m + f != max_players:
                    continue

                defenders = projections[projections['Position'] == 'D'].nlargest(d, 'ProjFPts')
                midfielders = projections[projections['Position'] == 'M'].nlargest(m, 'ProjFPts')
                forwards = projections[projections['Position'] == 'F'].nlargest(f, 'ProjFPts')

                current_combination = pd.concat([defenders, midfielders, forwards])
                current_score = current_combination['ProjFPts'].sum()

                if current_score > best_score:
                    best_combination = current_combination
                    best_score = current_score

    logging.info(f"Total Defenders: {len(best_combination[best_combination['Position'] == 'D'])}")
    logging.info(f"Total Midfielders: {len(best_combination[best_combination['Position'] == 'M'])}")
    logging.info(f"Total Forwards: {len(best_combination[best_combination['Position'] == 'F'])}")

    # Sort DataFrame by 'Pos' in the order 'D', 'M', 'F' and then by 'ProjFPts'
    best_combination.sort_values(by=['Position', 'ProjFPts'], key=lambda x: x.map({'D': 1, 'M': 2, 'F': 3}) if x.name == 'Position' else x, ascending=[True, False], inplace=True)
    best_combination.reset_index(drop=True, inplace=True)

    reserves = projections[~projections['Player'].isin(best_combination['Player'])].head(5).reset_index(drop=True)

    # Drop the 'Priority' column before returning
    best_combination.drop(columns=['Priority'], inplace=True, errors='ignore')
    reserves.drop(columns=['Priority'], inplace=True, errors='ignore')

    return best_combination, reserves

# Function that will get the average projected points for top 10 players across all managers within the same positional limits
def get_avg_proj_pts(players, projections):
    total_proj_pts = 0
    num_statuses = len(players['Status'].unique())

    for status in players['Status'].unique():
        top_10, _, top_10_proj_pts, _, _ = filter_by_status_and_position(players, projections, status)
        total_proj_pts += top_10_proj_pts

    average_proj_pts = round((total_proj_pts / num_statuses), 1)
    return average_proj_pts

def get_filtered_players(players, projections, status, projgs_value=None):
    """
    Filter players based on their status and optionally their ProjGS values.
    
    Args:
    - players (pd.DataFrame): DataFrame containing player information.
    - projections (pd.DataFrame): DataFrame containing projections for players.
    - status (str or list): Player status values to filter by.
    - projgs_value (int, optional): ProjGS value to filter players by. Defaults to None.
    
    Returns:
    - pd.DataFrame: Filtered player projections.
    - pd.DataFrame: Filtered player information.
    """
    # Check for the presence of 'Status' column and return early if not found.
    if 'Status' not in players.columns:
        logging.info("Warning: 'Status' column is not present in the players dataframe.")
        return pd.DataFrame(), pd.DataFrame()

    # Convert status to list if it's a string
    if isinstance(status, str):
        status = [status]

    # Filter players by their status
    filtered_players = players[players['Status'].isin(status)]

    # Optionally filter players by their ProjGS values
    if projgs_value is not None:
        filtered_players = filtered_players[filtered_players['ProjGS'] == projgs_value]

    if filtered_players.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Get a list of filtered player names
    player_list = filtered_players['Player'].unique().tolist()

    # Filter the projections DataFrame using the player list
    filtered_projections = projections[projections['Player'].isin(player_list)]

    # Prioritize players based on their ProjGS values
    filtered_projections['Priority'] = filtered_projections['ProjGS'].apply(lambda x: 0 if x == 0 else 1)
    filtered_projections.sort_values(by=['Priority', 'ProjFPts'], ascending=[False, False], inplace=True)

    return filtered_projections, filtered_players

def merge_and_drop_cols(df1, df2):
    """
    Merge two DataFrames and drop the specified columns.
    
    Args:
    - df1 (pd.DataFrame): First DataFrame to merge.
    - df2 (pd.DataFrame): Second DataFrame to merge.
    - cols_to_drop (list): List of columns to drop.
    
    Returns:
    - pd.DataFrame: Merged DataFrame.
    """
    merged_df = pd.merge(df1, df2, on='Player', how='left', suffixes=('', '_y'))
    # cols with '_y' suffix
    cols_to_drop = merged_df.filter(regex='_y').columns.tolist()
    st.write(f"Columns to drop: {cols_to_drop}")
    # drop cols with '_y' suffix
    merged_df.drop(columns=merged_df.filter(regex='_y'), inplace=True)
    return merged_df

def compute_aggregated_metrics(df):
    """
    Compute aggregated metrics for each unique Status.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the player data with 'Status', 'ProjFPts', and 'ROS Rank' columns.
    
    Returns:
    - pd.DataFrame: Aggregated metrics for each unique Status.
    """
    # Group by 'Status' and compute the mean of 'ProjFPts' and 'ROS Rank'
    aggregated = df.groupby('Status').agg({
        'ProjFPts': 'sum',
        'ROS Rank': 'mean'
    }).reset_index()
    
    return aggregated

# Initialize session states
if 'only_starters' not in st.session_state:
    st.session_state.only_starters = False

if 'lineup_clicked' not in st.session_state:
    st.session_state.lineup_clicked = False

if 'status' not in st.session_state:
    st.session_state.status = ""

def main():

    # Initialize session states
    # if 'only_starters' not in st.session_state:
    #     st.session_state.only_starters = False

    # if 'lineup_clicked' not in st.session_state:
    #     st.session_state.lineup_clicked = False

    # if 'status' not in st.session_state:
    #     st.session_state.status = ""
        
    # Adding construction banner or any other initial setups
    add_construction()

    custom_cmap = create_custom_sequential_cmap(*colors)

    mdlit(
    """### To get your optimal lineup head to -> @(https://www.fantrax.com/fantasy/league/d41pycnmlj3bmk8y/players;statusOrTeamFilter=ALL;pageNumber=1;positionOrGroup=SOCCER_NON_GOALIE;miscDisplayType=1) & follow the GIF below to populate and download the Players' data.
        """
        ) 
    with st.expander("How to download your data? (Click to expand instructions)"):
        add_vertical_space(2)
        local_gif(fx_gif)

    uploaded_file = st.file_uploader("Upload your player data", type="csv")

    if uploaded_file:
        center_running()
        with st.spinner('Loading data...'):
            players = pd.read_csv(uploaded_file)
            projections = load_csv(proj_csv)
            ros_ranks_data = load_csv(ros_ranks)

            # Renaming columns for consistency
            players.rename(columns={'Pos': 'Position'}, inplace=True, errors='ignore')
            projections.rename(columns={'Pos': 'Position'}, inplace=True, errors='ignore')

            # Merging data
            projections = pd.merge(projections, ros_ranks_data, how='left', on='Player', suffixes=('', '_y'))
            projections.drop(columns=projections.filter(regex='_y'), inplace=True)

            # Dropping unnecessary columns
            for df in [players, projections]:
                df.drop(columns=[col for col in ['Pos', '+/-'] if col in df.columns], inplace=True)

            projections['ROS Rank'].fillna(200, inplace=True)

            merged_df = merge_and_drop_cols(projections, players)

            projections = merged_df.copy()

            # print columns in merged_df
            st.write(f"Columns in merged_df: {merged_df.columns}")
            print(f"Columns in merged_df: {merged_df.columns}")

            # create a new dataframe grouped by Status and aggregate the ProjFPts and ROS Rank columns that we will use later to display the average projected points and average ROS Rank for each status
            grouped_status_df = compute_aggregated_metrics(merged_df)

            # reorder columns Player, Position, Team, ProjFPts, ProjGS, ROS Rank then rest of columns
            projections = projections[['Player', 'Position', 'Team', 'ProjFPts', 'ProjGPts', 'ProjGS', 'ROS Rank'] + [col for col in projections.columns if col not in ['Player', 'Position', 'Team', 'ProjFPts', 'ProjGPts', 'ProjGS', 'ROS Rank']]]

            debug_filtering(projections, players)

            players['Status'] = players['Status'].apply(lambda x: 'Waivers' if x.startswith('W (') else x)
            unique_statuses = sorted(players['Status'].unique())
            available_players = players[players['Status'].isin(['Waivers', 'FA'])]

            col_a, col_b = st.columns(2)

            with col_a:
                st.write("### 🛡️ Select your Fantasy team")
                new_status = st.selectbox('', unique_statuses)

                if new_status != st.session_state.status:
                    st.session_state.status = new_status
                    st.session_state.lineup_clicked = False  # Reset this state to trigger re-calculation

            with col_b:
                st.session_state.only_starters = st.checkbox('Only consider starters?')

            if st.button('🚀 Get my optimal lineup') or st.session_state.lineup_clicked:
                st.session_state.lineup_clicked = True
                status_list = [st.session_state.status]    

                col1, col2 = st.columns(2)

                with col1:
                    # status_list = [status] 
                    top_10, reserves, top_10_proj_pts, top_10_proj_pts_starters, roster = filter_by_status_and_position(players, projections, status_list)
                    # if top_10 contains player with ProjGS == 0, print Debug message and the top_10 dataframe and the reserves dataframe
                    # round the ProjFPts, ProjGPts columns to 1 decimal place
                    top_10 = top_10.round({'ProjFPts': 1, 'ProjGPts': 1})
                    # ensure ROS Rank is an integer
                    top_10['ROS Rank'] = top_10['ROS Rank'].astype(int)

                    st.write(f"### 🥇 {st.session_state.status} Best XI")
                    display_dataframe_pos(top_10)
                    st.write("### 🔄 Reserves")
                    display_dataframe_pos(reserves)

                with col2:
                    available_players = pd.merge(available_players, projections[['Player', 'ProjGS', 'ROS Rank']], on='Player', how='left')
                    top_10_waivers, reserves_waivers = filter_available_players_by_projgs(
                        available_players, projections, ['Waivers', 'FA'], 1 if st.session_state.only_starters else None
                    )
                    # round the ProjFPts, ProjGPts columns to 1 decimal place
                    top_10_waivers = top_10_waivers.round({'ProjFPts': 1, 'ProjGPts': 1})
                    # ensure ROS Rank is an integer
                    top_10_waivers['ROS Rank'] = top_10_waivers['ROS Rank'].astype(int)
                    st.write("### 🚀 Waivers & FA Best XI")
                    display_dataframe_pos(top_10_waivers)
                    st.write("### 🔄 Reserves")
                    display_dataframe_pos(reserves_waivers)

                average_proj_pts = get_avg_proj_pts(players, projections)

                col_c, col_d = st.columns(2)
                    
                with col_c:
                    with st.expander("Performance Metrics"):

                        avg_ros_of_top_fas = available_players.sort_values(by=['ROS Rank'], ascending=True).head(5)['ROS Rank'].mean()
                        average_proj_pts = get_avg_proj_pts(players, projections)
                        average_ros_rank_of_roster = round(roster['ROS Rank'].mean(), 1)
                        ros_rank_diff = round(average_ros_rank_of_roster - avg_ros_of_top_fas, 1)

                        # Compute the performance index for the top 10
                        top_10['performance_index'] = top_10['ProjFPts'] * top_10['ROS Rank']  # Multiply here to give higher score to players with higher ProjFPts and lower ROS Rank
                        performance_index_avg = top_10['performance_index'].mean()

                        # Compute the value score
                        value_score = performance_index_avg * ros_rank_diff

                        # Initialize the dataframe for value scores
                        value_score_df = pd.DataFrame(columns=['Status', 'Value Score', 'ProjFPts', 'ProjFPts*', 'Avg ROS Rank'])

                        for status in players['Status'].unique():
                            top_10, _, top_10_proj_pts, top_10_proj_pts_starters, _ = filter_by_status_and_position(players, projections, status)
                            average_ros_rank_of_roster = round(top_10['ROS Rank'].mean(), 1)
                            top_10['performance_index'] = top_10['ProjFPts'] * top_10['ROS Rank']
                            performance_index_avg = top_10['performance_index'].mean()
                            value_score_for_status = performance_index_avg * (average_ros_rank_of_roster - avg_ros_of_top_fas)

                            value_score_df.loc[len(value_score_df)] = [status, value_score_for_status, top_10_proj_pts_starters, average_ros_rank_of_roster]

                        # Normalize value score using MinMax scaling
                        min_value_score = value_score_df['Value Score'].min()
                        max_value_score = value_score_df['Value Score'].max()
                        value_score_df['Value Score'] = (value_score_df['Value Score'] - min_value_score) / (max_value_score - min_value_score)

                        # Rank the statuses based on the normalized value score
                        value_score_df.sort_values(by=['Value Score'], ascending=False, inplace=True)
                        value_score_df['Roster Rank'] = value_score_df['Value Score'].rank(method='dense', ascending=True).astype(int)
                        
                        # all metrics should be for the specific status_list selected
                        top_10_proj_pts = value_score_df[value_score_df['Status'].isin(status_list)]['ProjFPts'].values[0]
                        
                        top_10_proj_pts_starters = value_score_df[value_score_df['Status'].isin(status_list)]['ProjFPts*'].values[0]
                        st.write(f"### 🏆 {st.session_state.status} Performance Metrics")
                        # if top_10_proj_pts_starters is less than top_10_proj_pts, then add a delta using lambda
                        st.metric(label="🔥 Total Projected FPts", value=top_10_proj_pts, delta=round((top_10_proj_pts - top_10_proj_pts_starters), 1) if top_10_proj_pts_starters < top_10_proj_pts else None, delta_color="normal")

                        # if top_10_proj_pts_starters is less than top_10_proj_pts, then add a delta 
                        # st.metric(label="🔥 Total Projected FPts considering Projected Starts", value=top_10_proj_pts)

                        st.metric(label="🌟 Average XI ROS Rank", value=average_ros_rank_of_roster)
                        st.metric(label="📊 Value Score", value=value_score)
                        st.metric(label="💹 Avg Projected FPts of Best XIs across the Division", value=average_proj_pts, delta=round((top_10_proj_pts - average_proj_pts), 1))

                with col_d:
                    with st.expander("Value Score Rankings"):
                        # merge the value_score_df with the grouped_status_df
                        # value_score_df = pd.merge(value_score_df, grouped_status_df, on='Status', how='left', suffixes=('', '_grouped'))
                        # sort the value score dataframe by the value score column ascending
                        value_score_df.sort_values(by=['Value Score'], ascending=True, inplace=True)

                        st.dataframe(value_score_df.applymap(round_and_format), use_container_width=True)

            st.divider()

            if st.button('🔍 View all Projections'):
                projections = load_csv(proj_csv)
                st.dataframe(projections, use_container_width=True)


if __name__ == "__main__":
    # # Initialize session states
    # if 'only_starters' not in st.session_state:
    #     st.session_state.only_starters = False

    # if 'lineup_clicked' not in st.session_state:
    #     st.session_state.lineup_clicked = False

    custom_cmap = create_custom_cmap_cached(*colors)
    custom_divergent_cmap = create_custom_divergent_cmap_cached(*divergent_colors)

    main()
