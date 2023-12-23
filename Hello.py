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
from streamlit_extras.stoggle import stoggle


st.set_page_config(
    layout="wide",
    page_title="Draft Alchemy",
    page_icon=":soccer:",
    initial_sidebar_state="expanded",
)

# Set your upgraded API key
api_key = "60130162"


# Function to get all events from a season for a specific league
def get_events_from_season(league_id, season):
    version = "v1"
    base_url = f"https://www.thesportsdb.com/api/{version}/json/{api_key}/"
    url = f"{base_url}eventsseason.php"
    params = {"id": league_id, "s": season}
    st.write(f"Request URL: {url}")
    st.write(f"Request parameters: {params}")
    response = requests.get(url, params=params)
    st.write(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return None


# Function to get live scores
def get_live_scores(sport):
    version = "v2"
    base_url = f"https://www.thesportsdb.com/api/{version}/json/{api_key}/"
    url = f"{base_url}livescore.php"
    params = {"s": sport}
    st.write(f"Request URL: {url}?s={sport}")
    response = requests.get(url, params=params)
    st.write(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch live scores: {response.status_code}")
        return None


# Function to get event statistics by event ID
def get_event_statistics(event_id):
    version = "v1"
    base_url = f"https://www.thesportsdb.com/api/{version}/json/{api_key}/"
    url = f"{base_url}lookupeventstats.php"
    params = {"id": event_id}
    full_url = f"{url}?id={event_id}"
    st.write(f"Request URL: {full_url}")  # Display the full URL in the app
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch event statistics: {response.status_code}")
        return None


def main():
    add_construction()

    st.info(
        "This is a work in progress. None of the links on this page work yet. Use the sidebar to navigate to the other pages."
    )

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

    # Streamlit UI components to trigger API requests and display data
    st.title("TheSportsDB API Data")

    # Example usage of the get_events_from_season function
    league_id = "4328"  # English Premier League ID, for example
    season = "2023-2024"
    if st.button("Get Season Events"):
        events_data = get_events_from_season(league_id, season)
        if events_data and 'events' in events_data:
            game_weeks = sorted({event['intRound'] for event in events_data['events'] if event['intRound']})
            selected_week = st.selectbox("Select Game Week", options=game_weeks)

            matches = [event for event in events_data['events'] if event['intRound'] == selected_week]
            for match in matches:
                match_label = f"{match['strEventAlternate']}"
                stoggle(match_label, f"Event ID: {match['idEvent']}, Venue: {match['strVenue']}, Date: {match['dateEvent']}")
                
                if st.button(f"Get Stats for {match_label}", key=match['idEvent']):
                    event_stats = get_event_statistics(match['idEvent'])
                    if event_stats:
                        st.write(event_stats)
                    else:
                        st.write("No statistics available for this event.")

        else:
            st.write("No data available for the selected season.")

    # Example usage of the get_live_scores function
    sport = "Soccer"
    if st.button("Get Live Scores"):
        live_scores_data = get_live_scores(sport)
        if live_scores_data:
            st.write("Live Scores Data:")
            st.write(live_scores_data)
        else:
            st.write("No live scores available.")


if __name__ == "__main__":
    main()
