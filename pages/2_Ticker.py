import pandas as pd
import streamlit as st

# Read the CSV file
df = pd.read_csv('/Users/hogan/Library/CloudStorage/Dropbox/Mac/Documents/GitHub/st_fantrax_dashboard/data/specific-csvs/fdr_best.csv')

# Create a color-coded table
colors = {
    'ARS': 'red',
    'AVL': 'purple',
    'BOU': 'red',
    'BRF': 'green',
    'BHA': 'blue',
    'BUR': 'claret',
    'CHE': 'blue',
    'CRY': 'blue',
    'EVE': 'blue',
    'FUL': 'black',
    'LIV': 'red',
    'LUT': 'orange',
    'MCI': 'lightblue',
    'MUN': 'red',
    'NEW': 'black',
    'NOT': 'red',
    'SHU': 'red',
    'TOT': 'white',
    'WHU': 'claret',
    'WOL': 'gold'
}

# Display the table with color-coded cells
st.write('Premier League Teams Upcoming Fixtures')
st.write('')

# Create a new DataFrame to store the color-coded table
table_data = pd.DataFrame(columns=['Team', 'Fixture', 'Difficulty'])

# Iterate over each team in the original DataFrame
for index, row in df.iterrows():
    team = row['Team']
    fixtures = row[2:]  # Exclude the first two columns (Rank and Team)
    
    # Iterate over each fixture for the team
    for i, fixture in enumerate(fixtures):
        opponent = fixture.split(' ')[0]  # Extract the opponent code
        is_home = fixture.islower()  # Check if it's a home fixture
        
        # Determine the difficulty based on the numeric value
        difficulty = int(fixture.split('(')[1].split(')')[0])
        
        # Create a color-coded cell based on the difficulty
        cell = f'<font color="{colors[opponent]}">{opponent}</font>'
        
        # Add the fixture to the table data
        table_data = table_data.append({'Team': team, 'Fixture': cell, 'Difficulty': difficulty}, ignore_index=True)

# Sort the table data by difficulty in descending order
table_data = table_data.sort_values(by='Difficulty', ascending=False)

# Display the table
st.table(table_data)

