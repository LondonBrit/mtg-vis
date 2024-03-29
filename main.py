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
import json

##---CONSTANTS---##

API_URL = "https://api.scryfall.com/bulk-data"

def main():
    df = download_latest_data()
    print(df.head())
    pass

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

if __name__ == "__main__":
    main()