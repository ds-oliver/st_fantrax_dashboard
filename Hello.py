import streamlit as st

from functions import add_construction

st.set_page_config(
    layout="wide"
)

def main():
    
    add_construction()

    st.info('This is a work in progress. None of the links on this page work yet. Use the sidebar to navigate to the other pages.')

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
