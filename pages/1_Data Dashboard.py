import os
import sys
import logging
import pandas as pd
import warnings
import streamlit as st

# import plottable
# from plottable import Table
import mplsoccer
from mplsoccer import Radar, FontManager, PyPizza

# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime
# import uuid
# import streamlit_extras
# from streamlit_extras.dataframe_explorer import dataframe_explorer
# from markdownlit import mdlit
# from streamlit_extras.metric_cards import style_metric_cards
# from streamlit_extras.stylable_container import stylable_container
from streamlit_option_menu import option_menu
import time
import numpy as np

# import mplsoccer
import matplotlib.pyplot as plt

# import matplotlib.colors
from matplotlib.font_manager import FontProperties
from highlight_text import fig_text

# from mplsoccer import Bumpy, FontManager, add_image
from urllib.request import urlopen
from PIL import Image

# import plotly.figure_factory as ff
# import plotly.express as px
import altair as alt
import plotly.graph_objects as go


from constants import simple_colors, divergent_colors
from files import new_matches_data, ros_data
from functions import (
    load_css,
    add_construction,
    create_custom_cmap,
    create_custom_divergent_cmap,
    style_dataframe_custom,
    round_and_format,
    add_datadump_info,
)

TEXT_COLOR = "#fefae0"
BG_COLOR = "#370617"

# Set up relative path for the log file
current_directory = os.path.dirname(__file__)
log_file_path = os.path.join(current_directory, "info_streamlit_app_logs.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

st.set_page_config(page_title="Draft Alchemy", page_icon=":soccer:", layout="wide")

# load_css()


def load_csv_file(csv_file):
    return pd.read_csv(csv_file)


warnings.filterwarnings("ignore")

scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
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
    df = pd.read_csv(
        csv_file, index_col=0 if "Unnamed: 0" in pd.read_csv(csv_file).columns else None
    ).applymap(round_and_format)

    # Check if set_index_cols is provided
    if set_index_cols:
        # Check if all columns in set_index_cols exist in the DataFrame
        missing_cols = [col for col in set_index_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Columns {missing_cols} not found in DataFrame. Cannot set as index.\nDataFrame columns: {df.columns}"
            )

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
def display_dataframe(
    df,
    title,
    simple_colors,
    divergent_colors,
    info_text=None,
    upper_info_text=None,
    drop_cols=[],
):
    df = df.copy()

    df = df.drop(
        columns=[col for col in drop_cols if col in df.columns], errors="ignore"
    )

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
        styled_df = style_dataframe_custom(
            df,
            columns_to_keep,
            custom_cmap=custom_cmap,
            custom_divergent_cmap=custom_divergent_cmap,
            inverse_cmap=False,
            is_percentile=False,
        )
        st.dataframe(
            df[columns_to_keep].style.apply(lambda _: styled_df, axis=None),
            use_container_width=True,
            height=height,
        )
        logging.info(f"{title} Dataframe head: {df.head()}")
        logging.info(f"{title} Dataframe tail: {df.tail()}")
        if info_text:
            st.info(info_text)
    except Exception as e:
        logging.error(f"Error styling the {title} dataframe: {e}")
        st.error(f"Error styling the {title} dataframe: {e}")


def set_index_based_on_radio_button(df, widget_key, df_name="DataFrame"):
    """
    Set DataFrame index based on a Streamlit radio button.

    Parameters:
        df (pd.DataFrame): DataFrame to modify.
        widget_key (str): Unique key for the radio widget.

    Returns:
        pd.DataFrame: DataFrame with "Player" set as index if radio button is ticked.
    """
    set_index_option = st.radio(
        f"Would you like to set 'Player' as the {df_name} table index?",
        ("No", "Yes"),
        key=widget_key,
    )
    if set_index_option == "Yes":
        if "Player" in df.columns:
            df.set_index("Player", inplace=True)
        else:
            st.warning("The DataFrame does not have a 'Player' column.")
    return df


def clear_cache_button():
    """
    Adds a button to the Streamlit sidebar to clear the cache.
    """
    if st.sidebar.button("Clear cache"):
        st.cache_data.clear()
        st.rerun()


@st.cache_data
def get_sell_high_players(df, head=50):
    # create copy
    df = df.copy()
    print("printing df columns")
    print(df.columns)
    # sort by overperformance
    if "Overperformance" in df.columns and "HeatStreak" in df.columns:
        # get subset of data where Overperformance is greater than 0
        df = df[df["Overperformance"] > 0]
        # convert Overperformance to int and HeatStreak to float
        df["Overperformance"] = df["Overperformance"].astype(int)
        df["HeatStreak"] = df["HeatStreak"].astype(float).apply(lambda x: round(x, 2))
        # OvpAgg should be Overperformance * HeatStreak
        df["OvpAgg"] = round((df["Overperformance"] * df["HeatStreak"]), 2)
        # sort by OvpAgg
        df = df.sort_values(by=["OvpAgg"], ascending=False).head(head)
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
        date_created = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        return date_created
    else:
        return "File does not exist."


def display_date_of_update(date_of_update, title="Last Data Refresh"):
    return st.write(f"{title}: {date_of_update}")


# def plot_bumpy_chart(df, x_column, y_column, label_column, highlight_dict=None, **kwargs):
#     # Check if the required columns exist in the DataFrame
#     if not all(col in df.columns for col in [x_column, y_column, label_column]):
#         raise ValueError("The specified columns do not exist in the DataFrame.")

#     # Create lists for x and y axes
#     x_list = sorted(df[x_column].unique())
#     y_list = df[label_column].unique().tolist()

#     # Check if x_list and y_list are empty
#     if not x_list or not y_list:
#         raise ValueError("x_list or y_list is empty. Cannot plot an empty chart.")

#     # Create a dictionary of values for plotting
#     values = {}
#     for player in y_list:
#         player_df = df[df[label_column] == player]
#         values[player] = player_df[y_column].tolist()

#     # Check if values dictionary is empty
#     if not values:
#         raise ValueError("No data to plot.")

#     # Create the Bumpy object and plot the data
#     bumpy = Bumpy(**kwargs)
#     bumpy.plot(x_list, y_list, values, highlight_dict=highlight_dict)

#     plt.yticks(range(len(y_list)), y_list)

#     st.pyplot(plt.gcf())


def plot_bumpy_chart(
    df,
    x_column,
    y_column,
    label_column,
    highlight_dict=None,
    text_color="white",
    bg_color="black",
    **kwargs,
):
    # log columns to check
    st.write(f"Columns in df: {df.columns}")

    # Check if the required columns exist in the DataFrame
    if not all(col in df.columns for col in [x_column, y_column, label_column]):
        raise ValueError("The specified columns do not exist in the DataFrame.")

    # Sort DataFrame based on x_column and calculate rankings based on y_column
    df = df.sort_values(by=[x_column, y_column], ascending=[True, False])
    df["Rank"] = df.groupby(x_column)[y_column].rank(method="min", ascending=False)

    # Fill forward any missing rankings (if a player is missing for a particular game week)
    df["Rank"] = df.groupby(label_column)["Rank"].fillna(method="ffill")

    # Create lists for x and y axes
    x_list = sorted(df[x_column].unique())
    y_list = np.arange(1, df["Rank"].max() + 1).tolist()  # Rankings start from 1

    # Create a dictionary of values for plotting
    values = {}
    for player in df[label_column].unique():
        player_df = df[df[label_column] == player]
        rankings = (
            player_df.set_index(x_column)["Rank"].reindex(x_list).fillna(0).tolist()
        )
        values[player] = rankings

    # Instantiate the Bumpy object
    bumpy = Bumpy(
        scatter_color=bg_color,
        line_color="#252525",
        ticklabel_size=14,
        label_size=18,
        scatter_primary="D",
        show_right=True,
        plot_labels=True,
        alignment_yvalue=0.1,
        alignment_xvalue=0.065,
        **kwargs,
    )

    # Create the bumpy chart plot
    fig, ax = bumpy.plot(
        x_list,
        y_list,
        values,
        secondary_alpha=0.2,
        highlight_dict=highlight_dict,
        upside_down=True,  # <--- to flip the y-axis
        x_label="GW",
        y_label=y_column,  # label name
        lw=2.5,
        ylim=(1, max(y_list)),  # Explicitly set y-axis limits
    )

    # Font properties
    font_bold = FontProperties()
    font_bold.set_weight("bold")

    # Title
    TITLE = "Bumpy Chart Example:"
    fig.text(0.09, 0.95, TITLE, size=29, color=text_color, fontproperties=font_bold)

    # Subtitle with highlighted text
    SUB_TITLE = "A comparison between " + ", ".join(
        [f"<{player}>" for player in highlight_dict.keys()]
    )
    highlight_colors = [{"color": color} for color in highlight_dict.values()]

    fig_text(
        0.09,
        0.9,
        SUB_TITLE,
        color=text_color,
        highlight_textprops=highlight_colors,
        size=18,
        fig=fig,
        fontproperties=font_bold,
    )

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_percentile_bumpy_chart(
    df,
    label_column,
    metrics,
    highlight_dict=None,
    text_color="white",
    bg_color="black",
    **kwargs,
):
    # Calculate the percentile ranks for each player in the selected metrics
    for metric in metrics:
        df[f"{metric} Percentile"] = df[metric].rank(pct=True) * 100

    # Create lists for x and y axes
    x_list = [f"{metric}\nPercentile" for metric in metrics]
    y_list = np.linspace(0, 100, 11).astype(int).tolist()

    # Create a dictionary of values for plotting
    values = {}
    for player in df[label_column].unique():
        player_df = df[df[label_column] == player]
        player_values = [
            player_df[f"{metric} Percentile"].values[0] for metric in metrics
        ]
        values[player] = player_values

    # Instantiate the Bumpy object
    bumpy = Bumpy(
        rotate_xticks=0,
        ticklabel_size=23,
        label_size=28,
        scatter="value",
        show_right=True,
        alignment_yvalue=0.15,
        alignment_xvalue=0.06,
        **kwargs,
    )

    # Create the bumpy chart plot
    fig, ax = bumpy.plot(
        x_list,
        y_list,
        values,
        secondary_alpha=0.05,
        highlight_dict=highlight_dict,
        figsize=(20, 12),
        upside_down=True,
        x_label="Metrics",
        y_label="Percentile Rank",
        ylim=(0, 100),
        lw=2.5,
    )

    # Adjust plot position
    fig.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)

    # Font properties
    font_bold = FontProperties()
    font_bold.set_weight("bold")

    # Title
    TITLE = "Player Percentile Rank Comparison"
    fig.text(0.02, 0.98, TITLE, size=34, color=text_color, fontproperties=font_bold)

    # Subtitle with highlighted text
    if highlight_dict:
        SUB_TITLE = ", ".join([f"<{player}>" for player in highlight_dict.keys()])
        highlight_colors = [{"color": color} for color in highlight_dict.values()]

        fig_text(
            0.02,
            0.93,
            SUB_TITLE,
            color=text_color,
            highlight_textprops=highlight_colors,
            size=28,
            fig=fig,
            fontproperties=font_bold,
        )

    # Display the plot in Streamlit
    st.pyplot(fig)


relevant_stats = [
    "FPTS",
    "G",
    "Ghost Points",
    "Negative Fpts",
    "KP",
    "AT",
    "SOT",
    "TKW",
    "DIS",
    "YC",
    "RC",
    "ACNC",
    "INT",
    "CLR",
    "COS",
    "BS",
    "AER",
    "PKM",
    "PKD",
    "OG",
    "GAO",
    "CS",
]

# def compare_players(player_1_name="Erling Haaland", player_2_name="Mohamed Salah", player_stats_df):
#     player_1_stats = player_stats_df[player_stats_df['Player'] == player_1_name].iloc[0]
#     player_2_stats = player_stats_df[player_stats_df['Player'] == player_2_name].iloc[0]

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(player_1_stats['image_url'])  # Assuming there's a column with image URLs
#         st.subheader(player_1_name)
#         for stat in relevant_stats:
#             st.metric(label=stat, value=player_1_stats[stat])

#     with col2:
#         st.image(player_2_stats['image_url'])  # Assuming there's a column with image URLs
#         st.subheader(player_2_name)
#         for stat in relevant_stats:
#             st.metric(label=stat, value=player_2_stats[stat])

#     # Create a side-by-side bar chart for comparison
#     for stat in relevant_stats:
#         fig = go.Figure(data=[
#             go.Bar(name=player_1_name, x=[stat], y=[player_1_stats[stat]]),
#             go.Bar(name=player_2_name, x=[stat], y=[player_2_stats[stat]])
#         ])
#         # Change the bar mode
#         fig.update_layout(barmode='group')
#         st.plotly_chart(fig, use_container_width=True)


def compare_players_radar(
    player_stats_df, player_1_name, player_2_name, stats_to_include
):
    # Ensure stats_to_include is not empty
    if not stats_to_include:
        raise ValueError("No statistics provided for the radar chart.")

    # Filter the dataframe for the two players
    player_1_stats = (
        player_stats_df[player_stats_df["Player"] == player_1_name][stats_to_include]
        .values.flatten()
        .tolist()
    )
    player_2_stats = (
        player_stats_df[player_stats_df["Player"] == player_2_name][stats_to_include]
        .values.flatten()
        .tolist()
    )

    # Calculate the min and max values for each stat to define the range for the radar chart
    mins = player_stats_df[stats_to_include].min().values.flatten().tolist()
    maxs = player_stats_df[stats_to_include].max().values.flatten().tolist()

    # Instantiate Radar class with params, min_range, and max_range
    radar = Radar(params=stats_to_include, min_range=mins, max_range=maxs)

    # Define a figure
    fig, ax = radar.plot_radar(
        ranges=zip(mins, maxs),
        params=stats_to_include,
        values=[player_1_stats, player_2_stats],
        radar_color=["#1A78CF", "#FF920B"],
        alpha=0.25,
        compare=True,
    )
    # Add legends and titles
    ax.set_title(f"{player_1_name} vs {player_2_name}", size=20)
    plt.legend(labels=[player_1_name, player_2_name], loc="upper right", fontsize=12)

    # Show the radar chart
    st.pyplot(fig)


def create_scoring_distplot(pilot_scoring_all_gws_data, use_container_width: bool):
    # Assuming 'pilot_scoring_all_gws_data' is your DataFrame with the scoring data
    # Ensuring the scoring columns are numeric and clean
    pilot_scoring_all_gws_data["Default FPTS"] = pd.to_numeric(
        pilot_scoring_all_gws_data["Default FPTS"], errors="coerce"
    )
    pilot_scoring_all_gws_data["FPTS"] = pd.to_numeric(
        pilot_scoring_all_gws_data["FPTS"], errors="coerce"
    )
    pilot_scoring_all_gws_data.dropna(subset=["Default FPTS", "FPTS"], inplace=True)

    # Create a long-form DataFrame suitable for Altair
    long_form = pilot_scoring_all_gws_data.melt(
        value_vars=["Default FPTS", "FPTS"],
        var_name="Scoring System",
        value_name="Score",
    )

    # Altair Chart
    chart = (
        alt.Chart(long_form)
        .mark_bar(opacity=0.3, binSpacing=0)
        .encode(
            alt.X("Score:Q", bin=alt.Bin(maxbins=100)),
            alt.Y("count()", stack=None),
            alt.Color("Scoring System:N"),
        )
    )

    # Display the chart in a Streamlit application
    st.altair_chart(chart, use_container_width=use_container_width)


from matplotlib.font_manager import FontProperties


def create_pizza_chart(player_data, player_name, params, slice_colors, text_colors):
    # Check if the DataFrame is empty
    if player_data.empty:
        st.error(f"No data available for player {player_name}.")
        return

    # Extract player's stats
    player_values = [
        player_data[stat].values[0]
        if stat in player_data.columns
        and np.issubdtype(player_data[stat].dtype, np.number)
        and not np.isnan(player_data[stat].values[0])
        else 0
        for stat in params
    ]

    st.write(player_values)

    # Print the data types of the columns
    st.write(player_data[params].dtypes)

    # Check if the length of params, slice_colors, and text_colors are equal
    if not (len(params) == len(slice_colors) == len(text_colors)):
        st.error("The lengths of params, slice_colors, and text_colors do not match.")
        return

    # Instantiate PyPizza class
    baker = PyPizza(
        params=params,  # list of parameters
        background_color="#222222",  # background color
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,  # linewidth for straight lines
        last_circle_color="#000000",  # color for last line
        last_circle_lw=1,  # linewidth of last circle
        other_circle_lw=0,  # linewidth for other circles
        inner_circle_size=20,  # size of inner circle
    )

    # Plot pizza
    fig, ax = baker.make_pizza(
        player_values,  # list of values
        figsize=(8, 8.5),  # adjust the figsize according to your need
        color_blank_space="same",  # use the same color to fill blank space
        slice_colors=slice_colors,  # color for individual slices
        value_colors=text_colors,  # color for the value-text
        value_bck_colors=slice_colors,  # color for the blank spaces
        blank_alpha=0.4,  # alpha for blank-space colors
        kwargs_slices=dict(edgecolor="#000000", zorder=2, linewidth=1),
        kwargs_params=dict(color="#F2F2F2", fontsize=11, va="center"),
        kwargs_values=dict(
            color="#F2F2F2",
            fontsize=11,
            zorder=3,
            bbox=dict(
                edgecolor="#000000",
                facecolor="cornflowerblue",
                boxstyle="round,pad=0.2",
                lw=1,
            ),
        ),
    )

    # Add texts and titles
    fig.text(
        0.515,
        0.975,
        f"{player_name} - Performance",
        size=16,
        ha="center",
        color="#F2F2F2",
    )

    # Display the chart in Streamlit
    st.pyplot(fig)


def plot_grouped_bar_chart(df):
    # if FPTS exists rename to Pilot FPTS
    if "FPTS" in df.columns:
        df = df.rename(columns={"FPTS": "Pilot FPTS"})
    # Create traces for each scoring system with new colors
    trace1 = go.Bar(
        x=df["Player"],
        y=df["Default FPTS"],
        name="Default Scoring",
        marker=dict(color="#1f77b4"),  # Example of a blue color in hex
    )
    trace2 = go.Bar(
        x=df["Player"],
        y=df["Pilot FPTS"],
        name="New Scoring",
        marker=dict(color="#ff7f0e"),  # Example of an orange color in hex
    )

    # Arrange the traces to form a grouped bar chart
    data = [trace1, trace2]
    layout = go.Layout(
        barmode="group",
        title="Comparison of Scoring Systems",
        xaxis=dict(title="Players"),
        yaxis=dict(title="Fantasy Points"),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def main():
    epl = Image.open(
        urlopen(
            "https://raw.githubusercontent.com/andrewRowlinson/mplsoccer-assets/main/epl.png"
        )
    )

    data_path = "data/display-data/final"

    pilot_scoring_data_path = "data/display-data/final/pilot"

    # initialize logging

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

    recent_gw_players_df = load_csv_file_cached(f"{data_path}/recent_gw_data.csv")
    grouped_players_df = load_csv_file_cached(f"{data_path}/grouped_player_data.csv")
    all_gws_df = load_csv_file_cached(f"{data_path}/all_gws_data.csv")

    team_df = load_csv_file_cached(f"{data_path}/for_team.csv", set_index_cols=["team"])
    team_pos_df = load_csv_file_cached(
        f"{data_path}/d_detail_bypos_forteam.csv", set_index_cols=["team", "position"]
    )
    vs_team_df = load_csv_file_cached(
        f"{data_path}/vs_team.csv", set_index_cols=["opponent"]
    )

    # vs_team_pos_df = load_csv_file_cached(f'{data_path}/vs_team_pos_fbref.csv', set_index_cols=['opponent', 'position'])

    all_pos = load_csv_file_cached(
        f"{data_path}/all_pos.csv", set_index_cols=["position"]
    )
    ftx_pos_df = load_csv_file_cached(
        f"{data_path}/ftx_pos.csv", set_index_cols=["ftx_position"]
    )

    # d_df_pos = load_csv_file_cached(f'{data_path}/d_detail_bypos.csv', set_index_cols=['position'])
    # m_df_pos = load_csv_file_cached(f'{data_path}/m_detail_bypos.csv', set_index_cols=['position'])
    # f_df_pos = load_csv_file_cached(f'{data_path}/f_detail_bypos.csv', set_index_cols=['position'])

    home_team_byteam = load_csv_file_cached(
        f"{data_path}/home_team_byteam.csv", set_index_cols=["team"]
    )
    away_team_byteam = load_csv_file_cached(
        f"{data_path}/away_team_byteam.csv", set_index_cols=["team"]
    )

    # # load all big_six_df_teampos, newly_promoted_df_teampos, rest_teams_df_teampos, spotlight_teams_df_teampos
    big_six_teampos = load_csv_file_cached(
        f"{data_path}/big_six_teampos.csv", set_index_cols=["team", "position"]
    )
    newly_promoted_teampos = load_csv_file_cached(
        f"{data_path}/newly_promoted_teampos.csv", set_index_cols=["team", "position"]
    )
    rest_teams_teampos = load_csv_file_cached(
        f"{data_path}/rest_teams_teampos.csv", set_index_cols=["team", "position"]
    )
    spotlight_teams_teampos = load_csv_file_cached(
        f"{data_path}/spotlight_teams_teampos.csv", set_index_cols=["team", "position"]
    )

    recent_gw_data_team = load_csv_file_cached(
        f"{data_path}/recent_gw_data_agg_team.csv", set_index_cols=["Team"]
    )
    recent_gw_data_pos = load_csv_file_cached(
        f"{data_path}/recent_gw_data_agg_pos.csv", set_index_cols=["Position"]
    )
    recent_gw_data_teampos = load_csv_file_cached(
        f"{data_path}/recent_gw_data_agg_team_pos.csv",
        set_index_cols=["Team", "Position"],
    )

    set_piece_studs = load_csv_file_cached(f"{data_path}/top_5_players_per_team.csv")
    set_piece_studs_teams = load_csv_file_cached(
        f"{data_path}/set_piece_stats_team.csv", set_index_cols=["team"]
    )

    # call get_sell_high_players
    sell_high_players = get_sell_high_players(recent_gw_players_df, head=50)

    ### PILOT SCORING DATA ###
    # load pilot scoring data
    # data/display-data/final/pilot/recent_gw_data.csv
    pilot_gw_data = load_csv_file_cached(
        f"{pilot_scoring_data_path}/recent_gw_data.csv"
    )
    pilot_grouped_players_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/grouped_player_data.csv"
    )
    pilot_all_gws_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/pilot_scoring_all_gws_data.csv"
    )

    pilot_team_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/for_team.csv", set_index_cols=["team"]
    )
    pilot_vs_team_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/vs_team.csv", set_index_cols=["opponent"]
    )
    pilot_team_pos_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/team_gran_pos.csv",
        set_index_cols=["team", "position"],
    )
    pilot_team_ftx_pos_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/team_ftx_pos.csv",
        set_index_cols=["team", "ftx_position"],
    )

    pilot_all_pos = load_csv_file_cached(
        f"{pilot_scoring_data_path}/all_pos.csv", set_index_cols=["position"]
    )
    pilot_ftx_pos_df = load_csv_file_cached(
        f"{pilot_scoring_data_path}/ftx_pos.csv", set_index_cols=["ftx_position"]
    )

    # get the most recent gameweek value
    recent_gw = all_gws_df["GW"].max()
    first_gw = all_gws_df["GW"].min()

    default_style = {
        "container": {"padding": "5!important", "background-color": "#08061c"},
        "icon": {"color": "#fefae0", "font-size": "25px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "left",
            "margin": "0px",
            "--hover-color": "#6d597a",
        },
        "nav-link-selected": {"background-color": "#370617"},
    }

    # Create a dictionary to map dataframe names to actual dataframes and info_text
    df_dict = {
        # recent gw data
        f"Recent GW Data (GW {recent_gw})": {
            "frames": [
                {
                    # players who played in the most recent gameweek
                    "title": f"Player Data (GW {recent_gw})",
                    "data": recent_gw_players_df,
                    "info_text": f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. GPR is a measure of Ghost Points over Total FPts; the higher the value the better a ghoster the player is. The overperformance metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW rolling sum of FPTS. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value.",
                },
                {
                    # players who played in the most recent gameweek by team
                    "title": f"GW {recent_gw} Team Data",
                    "data": recent_gw_data_team,
                    "info_text": f"Note: This table shows team-specific data for GW {recent_gw}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "arrow-clockwise",
        },
        f"All GWs Data (GW {first_gw} - {recent_gw})": {
            "frames": [
                {
                    # all players, all gameweeks
                    "title": f"Player Data",
                    "data": grouped_players_df,
                    "info_text": f"Note: This table will show the statistics earned by each respective player, across all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                    "upper_info_text": f"",
                },
                {
                    "title": f"Team Data",
                    "data": team_df,
                    "info_text": f"Note: This table shows team-specific data for all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Basic Positional Data",
                    "data": ftx_pos_df,
                    "info_text": f"Note: This table shows basic position-specific data for all gameweeks. These are the simple Fantrax positions including {', '.join(ftx_pos_df.index.get_level_values('ftx_position').unique().tolist())}. At this time we are looking at :orange[{recent_gw}] gameweeks of data.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Granular Positional Data",
                    "data": all_pos,
                    "info_text": f"Note: This table shows position-specific data for all gameweeks. At this time we are looking at :orange[{max(recent_gw_players_df['GW'])}] gameweeks of data.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "minecart-loaded",
        },
        # set piece data
        "Set Piece Data": {
            "frames": [
                {
                    "title": "Set Piece Studs",
                    "data": set_piece_studs,
                    "info_text": f"Note: This table shows the top 5 players per team for set piece statistics. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data.",
                },
                {
                    "title": "Set Piece Studs (Team)",
                    "data": set_piece_studs_teams,
                    "info_text": f"Note: This table shows the set piece statistics for each team. The table is sorted by a deadball specialist aggregate metric. At this time we are looking at {max(recent_gw_players_df['GW'])} gameweeks of data.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "fire",
        },
        "Granular Team Data": {
            "frames": [
                {
                    "title": f"Team Data (GW {first_gw} - {recent_gw})",
                    "data": team_df,
                    "info_text": f"Note: This table shows team-specific data for GW {recent_gw}.",
                    "drop_cols": ["gp", "gp (max)", "gp (mean)"],
                    "upper_info_text": f"Aggregated data is filtered first for players who played more than 45 minutes",  # Example columns to drop
                },
                {
                    "title": f"vsTeam Data (GW {first_gw} - {recent_gw})",
                    "data": vs_team_df,
                    "info_text": f"Note: This table shows statistical categories teams have conceeded GW {recent_gw}.",
                    "drop_cols": [
                        "gp",
                        "gp (max)",
                        "gp (mean)",
                    ],  # Example columns to drop
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Home Team Data (GW {first_gw} - {recent_gw})",
                    "data": home_team_byteam,
                    "info_text": f"Note: This table shows data team-specific data based on Home performances for GW {recent_gw}.",
                    "drop_cols": [
                        "gp",
                        "gp (max)",
                        "gp (mean)",
                    ],  # Example columns to drop
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Away Team Data (GW {first_gw} - {recent_gw})",
                    "data": away_team_byteam,
                    "info_text": f"Note: This table shows data team-specific data based on Away performances for GW {recent_gw}.",
                    "drop_cols": [
                        "gp",
                        "gp (max)",
                        "gp (mean)",
                    ],  # Example columns to drop
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "table",
        },
        "Teams Data by Status": {
            "frames": [
                {
                    "title": f"Spotlight",
                    "data": spotlight_teams_teampos,
                    "info_text": f"Note: This table shows team-specific data by 'Spotlight' teams which include {', '.join(spotlight_teams_teampos.index.get_level_values('team').unique().tolist())}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Newly Promoted",
                    "data": newly_promoted_teampos,
                    "info_text": f"Note: This table shows team-specific data by 'Newly Promoted' teams which include {', '.join(newly_promoted_teampos.index.get_level_values('team').unique().tolist())}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Rest of League",
                    "data": rest_teams_teampos,
                    "info_text": f"Note: This table shows team-specific data by 'Rest of League' teams which include {', '.join(rest_teams_teampos.index.get_level_values('team').unique().tolist())}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "bar-chart-steps",
        },
        # positional data
        "Granular Positional Data": {
            "frames": [
                {
                    "title": f"Positional Data (GW {first_gw} - {recent_gw})",
                    "data": all_pos,
                    "info_text": f"Note: This table shows position-specific data for GW {recent_gw}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
                {
                    "title": f"Team Positional Data (GW {first_gw} - {recent_gw})",
                    "data": team_pos_df,
                    "info_text": f"Note: This table shows team-specific position data for GW {recent_gw}.",
                    "upper_info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                },
            ],
            "icon": "gem",
        },
        "Fantasy Insights": {
            "frames": [
                {
                    "title": f"Sell-High Candidates",
                    "data": sell_high_players,
                    "info_text": f"Note: This table shows top 50 players who are currently overperforming their RoS value and are on a good run of form.",
                }
            ],
            "icon": "lightbulb",
        },
        "Pilot Scoring Data": {
            "frames": [
                {
                    # players who played in the most recent gameweek by team
                    "title": f"All GWs Data (GW {first_gw} - {recent_gw}) | **:orange[Pilot Scoring]**",
                    "data": pilot_grouped_players_df,
                    "info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                    "drop_cols": [
                        "gp",
                        "gp (max)",
                        "gp (mean)",
                        "GW",
                    ],  # Example columns to drop
                    "upper_info_text": f"The table below shows {recent_gw} matches.",
                },
                {
                    # for_team
                    "title": f"Team Data (GW {first_gw} - {recent_gw}) | **:orange[Pilot Scoring]**",
                    "data": pilot_team_df,
                    "info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                    "upper_info_text": f"The table below shows {recent_gw} matches.",
                },
                {  # ftx_pos
                    "title": f"Basic Positional Data (GW {first_gw} - {recent_gw}) | **:orange[Pilot Scoring]**",
                    "data": pilot_ftx_pos_df,
                    "info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                    "upper_info_text": f"The table below shows {recent_gw} matches.",
                },
                {  # position
                    "title": f"Positional Data (GW {first_gw} - {recent_gw}) | **:orange[Pilot Scoring]**",
                    "data": pilot_all_pos,
                    "info_text": f"Aggregated data is filtered to include only players who played more than 45 minutes",
                    "upper_info_text": f"The table below shows {recent_gw} matches.",
                },
                {
                    # players who played in the most recent gameweek
                    "title": f"Player Data (GW {recent_gw}) | **:orange[Pilot Scoring]**",
                    "data": pilot_gw_data,
                    "info_text": f"Note: The above table is a subset of the full player data, filtered to show only players who have played in the most recent gameweek. GPR is a measure of Ghost Points over Total FPts; the higher the value the better a ghoster the player is. The 'Overperformance' metric is a simple difference of LiveRkOv (rank by Total FPts) less Ros Rank. A higher value will tell you the player is currently overperforming. HeatStreak is a 3 GW rolling sum of FPTS. If HeatStreak values are missing or null, it means there was insufficient data over the last 3 gameweeks to calculate a value.",
                },
            ],
            "icon": "moon-stars-fill",
        },
        "Charts": {
            "frames": [
                {
                    "title": "Percentile Plot",
                    "type": "percentile_bumpy",
                    "data": grouped_players_df,  # Example DataFrame
                    "label_column": "Player",
                    "metrics": [
                        "Attacking Stats",
                        "Defensive Stats",
                        "Duel Stats",
                        "Liability Stats",
                        "FPts/90",
                        "Ghosts/90",
                    ],
                }
            ],
            "icon": "graph-up",
        },
        # Adding a new chart type in the 'Charts' section
        "Scoring Distribution": {
            "frames": [
                {
                    "title": "Scoring System Comparison",
                    "type": "scoring_distplot",
                    "data": pilot_all_gws_df,  # Your DataFrame
                }
            ],
            "icon": "graph-up-fill",  # Choose an appropriate icon
        },
    }

    # Update the df_dict with a new entry for player comparison
    df_dict["Player Comparison"] = {
        "frames": [
            {
                "title": "Compare Players",
                "type": "player_comparison",
                # Assuming 'grouped_players_df' contains the stats for all players
                "data": grouped_players_df,
            }
        ],
        "icon": "graph-up",
    }

    df_dict["Player Stats Chart"] = {
        "frames": [
            {
                "title": "Player Stats Chart",
                "type": "player_pizza_chart",
                "data": grouped_players_df,
            }
        ],
        "icon": "chart-pie",
    }

    # List of the DataFrames to display based on the keys in the df_dict
    dfs_to_display = [(key, df_dict[key]["icon"]) for key in df_dict]

    # Define icons for each DataFrame
    dfs_keys = [df_key for df_key, _ in dfs_to_display]
    dfs_icons = [df_icon for _, df_icon in dfs_to_display]

    # Streamlit Option Menu for DataFrame selection
    with st.sidebar:
        selected_df_key = option_menu(
            "Select DataFrame",
            dfs_keys,
            icons=dfs_icons,
            menu_icon="list",
            styles=default_style,
        )

    # Conditionally display the selected DataFrame and info text
    if selected_df_key:
        st.toast("Loading data...")
        selected_frames = df_dict.get(selected_df_key, {}).get("frames", [])
        for frame in selected_frames:
            # if frame.get("type") == "percentile_bumpy":
            #     # Filter the DataFrame based on selected players
            #     available_players = (
            #         frame["data"][frame["label_column"]].unique().tolist()
            #     )

            #     # Create a Streamlit multi-select widget for selecting players
            #     selected_players = st.multiselect(
            #         "Select Players", available_players, default=available_players[:3]
            #     )  # Default to first 3 players

            #     # Automatically assign colors to the selected players
            #     colors = ["salmon", "cornflowerblue", "gold"]
            #     highlight_dict = {
            #         player: color for player, color in zip(selected_players, colors)
            #     }

            #     # Filter the DataFrame based on selected players
            #     filtered_df = frame["data"][
            #         frame["data"][frame["label_column"]].isin(selected_players)
            #     ]

            #     # Debug: Check if DataFrame is empty
            #     if filtered_df.empty:
            #         st.write("Filtered DataFrame is empty.")

            #     # Plot the percentile bumpy chart
            #     plot_percentile_bumpy_chart(
            #         filtered_df,
            #         frame["label_column"],
            #         frame["metrics"],
            #         highlight_dict=highlight_dict,
            #     )

            # elif frame.get("type") == "scoring_distplot":
            #     # Call your distribution plot function here
            #     # create_scoring_distplot(frame["data"], use_container_width=True)
            #     plot_grouped_bar_chart(frame["data"])

            if frame.get("type") == "player_comparison":
                # Logic for selecting players to compare
                all_players = frame["data"]["Player"].unique().tolist()
                player_1_name = st.selectbox(
                    "Select Player 1",
                    all_players,
                    index=all_players.index("Erling Haaland")
                    if "Erling Haaland" in all_players
                    else 0,
                )
                player_2_name = st.selectbox(
                    "Select Player 2",
                    all_players,
                    index=all_players.index("Mohamed Salah")
                    if "Mohamed Salah" in all_players
                    else 1,
                )

                # Define the stats you want to include in your radar chart
                stats_to_include = [
                    "FPTS",
                    "G",
                    "Ghost Points",
                    "Negative Fpts",
                    "KP",
                    "AT",
                ]

                if st.button("Compare"):
                    compare_players_radar(
                        frame["data"], player_1_name, player_2_name, stats_to_include
                    )

            elif frame.get("type") == "player_pizza_chart":
                # Logic to handle player selection for pizza chart
                all_players = frame["data"]["Player"].unique().tolist()
                selected_player = st.selectbox("Select Player", all_players)

                # Define the stats you want to include in your pizza chart
                stats_to_include = [
                    "FPTS",
                    "G",
                    "Ghost Points",
                    "Negative Fpts",
                    "KP",
                    "AT",
                ]

                # Define default slice and text colors
                default_slice_colors = [
                    "#1A78CF",
                    "#FF9300",
                    "#D70232",
                    "#F05B4F",
                    "#8A9B0F",
                    "#FFCD00",
                ]
                default_text_colors = [
                    "#FFFFFF",
                    "#000000",
                    "#FFFFFF",
                    "#FFFFFF",
                    "#000000",
                    "#000000",
                ]

                if st.button("Display"):
                    # Check if selected stats are in the dataframe
                    if not all(
                        stat in frame["data"].columns for stat in stats_to_include
                    ):
                        st.error("One or more selected stats are not in the DataFrame.")
                        continue

                    # Extract player values for the selected stats
                    player_data = frame["data"][
                        frame["data"]["Player"] == selected_player
                    ]

                    # Call the function to create and display the pizza chart
                    create_pizza_chart(
                        player_data,
                        selected_player,
                        stats_to_include,
                        default_slice_colors,
                        default_text_colors,
                    )

            # elif frame.get("type") == "player_comparison":
            #     # Logic for selecting players to compare
            #     all_players = frame["data"]["Player"].unique().tolist()
            #     player_1_name = st.selectbox("Select Player 1", all_players)
            #     player_2_name = st.selectbox("Select Player 2", all_players)

            #     if st.button("Compare"):
            #         compare_players(player_1_name, player_2_name, frame["data"])

            else:
                # ... (handling for other visualization and data types)
                display_dataframe(
                    frame["data"],
                    frame["title"],
                    simple_colors,
                    divergent_colors,
                    info_text=frame.get("info_text"),
                    upper_info_text=frame.get("upper_info_text"),
                    drop_cols=frame.get("drop_cols", []),
                )
    else:
        st.error(f"DataFrame '{selected_df_key}' not found where expected.")

    logging.info("Main function completed successfully")


if __name__ == "__main__":
    main()
