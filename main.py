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
from bokeh.models import BoxAnnotation, Label, Legend, LegendItem, WheelZoomTool, LabelSet, ColumnDataSource, DataTable, StringFormatter, NumberFormatter, IntEditor, NumberEditor, StringEditor, SelectEditor, DateFormatter, DateEditor, TableColumn
from bokeh.io import output_file, save
from bokeh.layouts import gridplot, column
from math import pi



##---CONSTANTS---##

API_URL = "https://api.scryfall.com/bulk-data"
CURRENT_YEAR = 2024

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
    df.loc[:, 'oracle_text'] = df['oracle_text'].fillna('')

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

    #show(p)
    save(p, filename="./mtg-vis/wordiness.html")

    return p


def are_wordy_cards_pushing_out_flavour_text(df):

# Sort by 'name' and 'released_at' so that the earliest print of a card is first
    df = df.sort_values(by=['name', 'released_at'])

# Drop duplicates keeping the first occurrence (which is the earliest release of a card)
    df_unique = df.drop_duplicates(subset='name', keep='first').copy()

# Create a year column for grouping
    df_unique['year'] = df_unique['released_at'].dt.year

# Group by set_name and year, and calculate the word count in oracle_text
    word_counts = df_unique.groupby(['set_name', 'year', 'flavor_text'])['oracle_text'].apply(lambda x: x.str.split().str.len()).reset_index(name='word_count')
    
# Create a new column 'has_flavour_text' that is 1 if 'flavour_text' is not NaN, and 0 otherwise
    df_unique['has_flavour_text'] = df_unique['flavor_text'].notnull().astype(int)

# Group by 'year' and calculate the total number of cards and the number of cards with flavour text
    total_cards_per_year = df_unique.groupby('year').size()
    cards_with_flavour_text_per_year = df_unique.groupby('year')['has_flavour_text'].sum()

# Create a DataFrame with the percentage of cards with flavour text
    summary_data = pd.DataFrame({
        'mean_word_count': word_counts.groupby('year')['word_count'].mean(),
        'percentage_has_flavour_text': cards_with_flavour_text_per_year / total_cards_per_year * 100
    }).reset_index()
    
# Rename the columns
    summary_data = summary_data.rename(columns={'word_count': 'mean_word_count', 'percentage_has_flavour_text': 'percentage_has_flavour_text'})

# Normalize the 'mean_word_count' column
    #summary_data['mean_word_count'] = (summary_data['mean_word_count'] - summary_data['mean_word_count'].min()) / (summary_data['mean_word_count'].max() - summary_data['mean_word_count'].min())

# Normalize the 'percentage_has_flavour_text' column
    #summary_data['percentage_has_flavour_text'] = (summary_data['percentage_has_flavour_text'] - summary_data['percentage_has_flavour_text'].min()) / (summary_data['percentage_has_flavour_text'].max() - summary_data['percentage_has_flavour_text'].min())

# Create a scatter plot to compare mean word count and sum of has flavour text
    p = figure(title="Word Count vs Flavour Text", x_axis_label="Mean Word Count", y_axis_label="% Has Flavour Text", 
               tools="pan,wheel_zoom,xbox_select,reset", sizing_mode="scale_width", height=800)

# Hide the axis values (ticks and labels)
    p.xaxis.major_label_text_font_size = '0pt'  # preferred method 
    p.yaxis.major_label_text_font_size = '0pt'
    p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
    p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
    p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
    p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks
    
# Add scatter plot
    p.scatter(summary_data['mean_word_count'], summary_data['percentage_has_flavour_text'], size=10, color='navy', alpha=0.8, marker="circle")

# Add a trend line
    slope, intercept = np.polyfit(summary_data['mean_word_count'], summary_data['percentage_has_flavour_text'], 1)
    x = np.linspace(summary_data['mean_word_count'].min(), summary_data['mean_word_count'].max(), 100)
    y = slope * x + intercept
    p.line(x, y, color='red', line_width=2)

# Add data labels to the scatter plot
    labels = LabelSet(x='mean_word_count', y='percentage_has_flavour_text', text='year', level='glyph',
                    x_offset=5, y_offset=5, source=ColumnDataSource(summary_data))
    p.add_layout(labels)

# Create a DataFrame with un-normalized values
    summary_data_unnormalized = pd.DataFrame({
        'mean_word_count': word_counts.groupby('year')['word_count'].mean(),
        'percentage_has_flavour_text': cards_with_flavour_text_per_year / total_cards_per_year
    }).reset_index()

# Create a ColumnDataSource from the un-normalized DataFrame
    source = ColumnDataSource(summary_data_unnormalized)

# Define the columns for the DataTable
    columns = [
        TableColumn(field="year", title="Year", editor=IntEditor(), formatter=NumberFormatter(format="0"), width=50),
        TableColumn(field="mean_word_count", title="Mean Word Count", editor=NumberEditor(step=0.1), formatter=NumberFormatter(format="0.00")),
        TableColumn(field="percentage_has_flavour_text", title="% Has Flavour Text", editor=NumberEditor(step=0.1), formatter=NumberFormatter(format="0.00%")),
    ]

# Create a DataTable
    data_table = DataTable(source=source, columns=columns, editable=True, index_position=-1)   


#Adjust the widths
    p.width = 1200
    data_table.width = 800

# Add the scatter plot and the DataTable to the layout
    layout = column(p, data_table)

# Show the layout
    return(layout)

def generate_flavour_text_html(df):
    flavour_text_layout = are_wordy_cards_pushing_out_flavour_text(df)
    output_file("flavour_text.html")
    save(flavour_text_layout, filename="./mtg-vis/flavour_text.html")
    #show(flavour_text_layout)
    pass


def main():
    df = download_latest_data()
    df = process_data(df)
    #wordiness_graph = how_wordy_are_cards(df,CURRENT_YEAR)
#Create a seperate page for the flavour text html
    generate_flavour_text_html(df)
    #grid = gridplot([[wordiness_graph], [ft_comparison_graph])
    #save(grid, filename="./mtg-vis/dashboard.html")
    

if __name__ == "__main__":
    main()