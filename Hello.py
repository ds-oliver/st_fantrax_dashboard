import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from PIL import Image
import numpy as np
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io
from functions import add_construction
from thesportsdb import events, leagues
import requests
import json
import os

st.set_page_config(
    layout="wide",
    page_title="Draft Alchemy",
    page_icon=":soccer:",
    initial_sidebar_state="expanded",
)

# Your unique API key

api_key = "60130162"

# Base URL for TheSportsDB API
base_url = "https://www.thesportsdb.com/api/v1/json/{}/".format(api_key)

# Function to fetch live scores for soccer
def fetch_live_scores():
    live_scores = events.leagueSeasonEvents(league_id="4328", season="2023")
    return live_scores


# Function to fetch all events from a particular season
def fetch_season_events():
    season_events = events.leagueSeasonEvents(league_id="4328", season="2023")
    return season_events

# Function to get all events from a season for a specific league
def get_events_from_season(league_id, season):
    url = base_url + "eventsseason.php"
    params = {"id": league_id, "s": season}
    response = requests.get(url, params=params)
    return response.json()

# Function to get live scores
def get_live_scores(sport):
    url = base_url + "livescore.php"
    params = {"s": sport}
    response = requests.get(url, params=params)
    return response.json()


def main():
    add_construction()

    st.info(
        "This is a work in progress. None of the links on this page work yet. Use the sidebar to navigate to the other pages."
    )

    # with st.sidebar:
    #     choose = option_menu("Toolkit Menu",
    #                             ["Home", "Optimal Lineup", "Matchup Projections", "Add/Drop Suggestions",
    #                             "Trade Calculator", "Fixture Difficulty Tracker", "GW Transaction Data", "Team Power Rankings", "Glossary", "Pick Diff Team"],
    #                             icons=['house', 'list-columns', 'lightning-charge', 'person-plus',
    #                                 'calculator', 'calendar-range', 'graph-up-arrow', 'list-ol', 'book', 'arrow-clockwise'],
    #                             menu_icon="menu-app", default_index=0)

    st.title("This is part of @draftalchemy | @ds-oliver FPL Data Science Project")
    st.write(
        """
    Welcome to the FPL Data Science Platform! Our platform provides access to comprehensive datasets including player statistics, match results, historical performance, and more. Analyze data from top leagues around the world or dive into specific player performance metrics.
    """
    )

    st.header("Features")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fantasy Data")
        st.write(
            "Analyze player performance over time, compare players, and identify trends."
        )
        # embed link "https://fx-dash.streamlit.app/Data_Dashboard"
        st.markdown(
            f'<a href="https://fx-dash.streamlit.app/Data_Dashboard" target="_blank"><input type="button" value="Go to data dashboard"></a>',
            unsafe_allow_html=True,
        )

    with col2:
        st.subheader("Current Season")
        st.write(
            "View player statistics for the current season, including live data during matches."
        )
        st.button("Explore Current Season")

    # with col3:
    #     st.subheader('Fantasy Tools')
    #     st.write('Use our predictive models to build optimal line-ups, simulate matches, and make informed decisions.')
    #     st.button('Explore Fantasy Tools')

    # st.header('Datasets')
    # st.write("""
    # Our platform provides access to comprehensive datasets including player statistics, match results, historical performance, and more. Analyze data from top leagues around the world or dive into specific player performance metrics.
    # """)

    # st.header('Get Started')
    # st.write("""
    # Navigate through our platform using the buttons above or the sidebar menu. Whether you're a soccer fan, fantasy manager, or data enthusiast, we have something for you!
    # """)

    # Streamlit UI components to trigger API requests and display data
    st.title("TheSportsDB API Data")

    # Fetch and display league season events
    league_id = "4328"  # Example league ID for English Premier League
    season = "2023"  # Example season
    if st.button("Get Season Events"):
        events_data = get_events_from_season(league_id, season)
        st.write("League Season Events Data:")
        st.write(events_data)

    # Fetch and display live scores
    sport = "Soccer"  # Example sport
    if st.button("Get Live Scores"):
        live_scores_data = get_live_scores(sport)
        st.write("Live Scores Data:")
        st.write(live_scores_data)


if __name__ == "__main__":
    main()
