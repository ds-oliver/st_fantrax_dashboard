# doc paths
import re
import os

# match reports 
# data/data_out/scraped_big5_data/pl_data/all_shots_all_20230820.csv data/data_out/scraped_big5_data/pl_data/full_season_matchreports_20230820.csv
# list comprehension that gets the file in 'data/projections/' where filename matches 'GW{is_number} Projections.csv' 
projections = [f'data/projections/{file}' for file in os.listdir('data/projections/') if re.match(r'GW\d+ Projections.csv', file)][0]

ros_ranks = [f'data/ros-data/{file}' for file in os.listdir('data/ros-data/') if re.match(r'Weekly ROS Ranks_GW\d+.csv', file)][0]


shots_data = 'data/data_out/scraped_big5_data/pl_data/all_shots_all_20240123.csv'
new_matches_data = 'data/fbref-data/full_season_matchreports_20240123.csv'
ros_data = 'data/ros-data/Weekly ROS Ranks_GW8.csv'

# files paths
data_out = 'data/data_out'

# data paths
scraped_big5_data = 'data/data_out/scraped_big5_data'

fx_gif = 'media/download_fantrax_data_demo.gif'

pl_2018_2023 = 'data/data_out/scraped_big5_data/pl_data/player_table_2018_to_2023.csv'