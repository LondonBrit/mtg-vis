import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.transforms as mtransforms
import matplotlib.style as style
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.stats import zscore
import requests
from datetime import datetime, timedelta
import os
import glob
import matplotlib.patches as patches
import json
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, Label, Legend, LegendItem, WheelZoomTool
from bokeh.io import output_file, save
from math import pi



##---CONSTANTS---##

API_URL = "https://api.scryfall.com/bulk-data"
CURRENT_YEAR = 2024

def main():
    df = download_latest_data()
    df = process_data(df)
    how_wordy_are_cards(df,CURRENT_YEAR)
    pass


# Define a function to concatenate oracle_text from card_faces
def concatenate_oracle_text(row):
    if isinstance(row['card_faces'], list):
        oracle_texts = [face['oracle_text'] for face in row['card_faces'] if 'oracle_text' in face]
        return ' '.join(oracle_texts)
    else:
        return row['oracle_text']


# Define a function to process the data
def process_data(df):
    # Keywords to exclude
    excluded_keywords = ["token", "playtest", "scheme", "planechase", "sticker", "art series", "alchemy", "anthology","minigames","MTG Arena"]

    # Filter out sets based on keywords
    df = df[~df['set_name'].str.lower().str.contains('|'.join(excluded_keywords))]

    # Handle NaN values in 'oracle_text'
    df['oracle_text'] = df['oracle_text'].fillna('')

    # Remove sets with fewer than 30 cards
    #filtered_sets = df['set_name'].value_counts()
    #valid_sets = filtered_sets[filtered_sets >= 30].index
    #df = df[df['set_name'].isin(valid_sets)]

    # Filter out cards with a rarity of 'special'
    df = df[df['rarity'] != 'special']

    # Filtering out rows where set_name starts with "un" (case-insensitive)
    #df = df[~df['set_name'].str.lower().str.startswith('un')]

    # Assuming 'released_at' is a string representing dates, convert to datetime
    df['released_at'] = pd.to_datetime(df['released_at'], errors='coerce')

    # Drop rows with NaT in 'released_at' if any conversion errors occurred
    df = df.dropna(subset=['released_at'])

    # Extract year from 'released_at' datetime column
    df['release_year'] = df['released_at'].dt.year

    # Filter out all data from 2025
    #df = df[df['release_year'] != 2025]

    # Concatenate oracle text
    df['oracle_text'] = df.apply(concatenate_oracle_text, axis=1)

    # Replace all instances of '\n' with a space character in 'oracle_text'
    df['oracle_text'] = df['oracle_text'].str.replace('\n', ' ')

    return df


# Function to download the default cards file
def download_file(download_uri, filename):
    response = requests.get(download_uri)
    if response.status_code == 200:
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded the new file: {filename}")
        return filename
    else:
        print("Failed to download file.")
        return None


def download_latest_data():
    # File choice
    file_id = "default"

    # Step 1: Check the directory for the 'default-cards' file
    file_pattern = f"{file_id}-cards*.json"  # Adjust pattern as needed
    file_list = glob.glob(file_pattern)
    latest_file = None

    # Determine the latest file based on the modification time
    for file in file_list:
        if latest_file is None or os.path.getmtime(file) > os.path.getmtime(latest_file):
            latest_file = file

    data_file = latest_file  # Assume the latest local file is the data file

    # If there's no latest file found or it's older than a week, download a new one
    if latest_file is None or (datetime.now() - datetime.fromtimestamp(os.path.getmtime(latest_file))) > timedelta(days=7):
        response = requests.get(API_URL)
        if response.status_code == 200:
            bulk_data_info = response.json()
            for entry in bulk_data_info.get("data", []):
                if entry["name"].lower() == f"{file_id} cards":
                    # Found the default cards entry, proceed to download
                    filename = entry["download_uri"].split('/')[-1]
                    data_file = download_file(entry["download_uri"], filename)
                    break
        else:
            print("Failed to retrieve data from the API")
    else:
        print(f"The latest data file available is '{data_file}' and it is not a week old yet.")

    # Ensure `data_file` is not None and is the latest file
    if data_file is None:
        print("No valid data file found or downloaded.")
    else:
        print(f"Using the data file: {data_file}")

    # At this point, `data_file` will have the name of the latest file

    # Load data from the file
    df= pd.read_json(data_file)

    return df

def how_wordy_are_cards(df,year):

    # Calculate average word count per set per year
    def count_words(string):
        return len(str(string).split())

    # Sort by 'name' and 'released_at' so that the earliest print of a card is first
    df = df.sort_values(by=['name', 'released_at'])

    # Drop duplicates keeping the first occurrence (which is the earliest release of a card)
    df_unique = df.drop_duplicates(subset='name', keep='first').copy()

    # Create a year column for grouping
    df_unique['year'] = df_unique['released_at'].dt.year

    # Group by set_name and year, and calculate the word count in oracle_text
    word_counts = df_unique.groupby(['set_name', 'year'])['oracle_text'].apply(lambda x: x.str.split().str.len()).reset_index(name='word_count')
    word_counts = word_counts.drop(columns=['level_2'])

    # Sort the dataframe by 'released_at' in ascending order
    word_counts = word_counts.sort_values(by='year')

    ## Next bit is just to give a view per year for every year before 2024

    # Filter out rows where 'year' is 2024
    word_counts_not_2024 = word_counts[word_counts['year'] != year]

    # Group by 'year' and calculate the mean word count
    mean_word_counts = word_counts_not_2024.groupby('year')['word_count'].mean().reset_index(name='mean_word_count')

    #change the index column to be set name
    mean_word_counts['index_column'] = mean_word_counts['year']

    # Filter rows where 'year' is 2024
    word_counts_2024 = word_counts[word_counts['year'] == year]

    #Group the sets together. Calculate the mean word count per set
    set_mean_word_count = word_counts_2024.groupby('set_name')['word_count'].mean().reset_index(name='mean_word_count')

    #change the index column to be set name
    set_mean_word_count['index_column'] = set_mean_word_count['set_name']

    # Concatenate the two DataFrames
    final_df = pd.concat([mean_word_counts, set_mean_word_count], ignore_index=True)

    # Output to file. This will create a new HTML file that subsequent commands will edit.
    output_file("wordiness.html")

    # Create a new plot with a title and axis labels
    p = figure(title="MTG: Average Word Count", x_axis_label='Year', y_axis_label='Average Word Count', 
            tools="pan,wheel_zoom,xbox_select,reset", sizing_mode="scale_width", height=200)
    
    # Set wheel zoom as the active scroll tool
    p.toolbar.active_scroll = p.select_one(WheelZoomTool)

    # Add a bar chart to the plot
    p.vbar(x=final_df.index, top=final_df['mean_word_count'], width=0.5, color='ForestGreen', alpha=0.6, legend_label='Average Word Count')
    
    # Add data labels to the bars
    for i in range(len(final_df)):
        label = Label(x=i-0.5, y=final_df['mean_word_count'].iloc[i], 
                    text='{:.2f}'.format(final_df['mean_word_count'].iloc[i]), 
                    text_font_size="12pt", text_color="#555555", angle=45*pi/180)
        p.add_layout(label)

    # Find the row where "index_column" is 2023
    start_row = final_df[final_df['index_column'] == year-1].index[0]

    # Calculate the start position and width of the rectangle
    start_pos = start_row - final_df.index[0]  # Subtract the first index to align with the x-axis ticks
    width = len(final_df.index) - start_pos

    # Add a shaded box
    box = BoxAnnotation(left=start_pos, right=start_pos+width, fill_color='lightblue', fill_alpha=0.5)
    p.add_layout(box)

    # Add a label to the top left inner corner of the box
    label = Label(x=start_pos+1, y=final_df['mean_word_count'].max(), text='2024 Sets', text_font_size='22pt')
    p.add_layout(label)

    # Set x-axis category labels
    p.xaxis.ticker = final_df.index
    p.xaxis.major_label_overrides = {i: str(year) for i, year in enumerate(final_df['index_column'])}
    p.xaxis.major_label_orientation = 45

    show(p)
    save(p, filename="./mtg-vis/wordiness.html")

    pass


if __name__ == "__main__":
    main()