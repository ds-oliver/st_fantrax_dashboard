import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import cv2
import pandas as pd
from st_aggrid import AgGrid
import plotly.express as px
import io 

from functions import add_construction

st.set_page_config(
    layout="wide",
    page_title="Draft Alchemy",
    page_icon=":soccer:",
    initial_sidebar_state="expanded"
)

def main():
    
    add_construction()

    st.info('This is a work in progress. None of the links on this page work yet. Use the sidebar to navigate to the other pages.')

    # with st.sidebar:
    #     choose = option_menu("Toolkit Menu",
    #                             ["Home", "Optimal Lineup", "Matchup Projections", "Add/Drop Suggestions",
    #                             "Trade Calculator", "Fixture Difficulty Tracker", "GW Transaction Data", "Team Power Rankings", "Glossary", "Pick Diff Team"],
    #                             icons=['house', 'list-columns', 'lightning-charge', 'person-plus',
    #                                 'calculator', 'calendar-range', 'graph-up-arrow', 'list-ol', 'book', 'arrow-clockwise'],
    #                             menu_icon="menu-app", default_index=0)
                    

    st.title('This is part of @draftalchemy | @ds-oliver FPL Data Science Project')
    st.write("""
    Welcome to the FPL Data Science Platform! Our platform provides access to comprehensive datasets including player statistics, match results, historical performance, and more. Analyze data from top leagues around the world or dive into specific player performance metrics.
    """)

    st.header('Features')
    col1, col2 = st.columns(2)

    with col1:
        st.subheader('Historical Data')
        st.write('Analyze player performance over time, compare players, and identify trends.')
        st.button('Explore Historical Data')

    with col2:
        st.subheader('Current Season')
        st.write('View player statistics for the current season, including live data during matches.')
        st.button('Explore Current Season')

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

if __name__ == "__main__":
    main()
