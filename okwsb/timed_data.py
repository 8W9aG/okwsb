"""The module for storing timed data."""
import json
import csv
from io import StringIO
import time
import os
import glob
import random

import requests
from dateutil.parser import parse


STOCK_TICKERS = ["IBM", "AAPL", "MSFT"]
STOCKS_KEY = "stocks"
TRAINING_DATA_FOLDER = ".training_data"


def datetime_to_key(datetime):
    """Convert a datetime to a key."""
    return datetime.strftime("%Y_%m_%d")


def load_alphavantage_time_series_intraday_extended(stock_ticker, avslice, timed_data, key):
    """Load the alphavantage time series intraday extended data."""
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={stock_ticker}&interval=1min&slice={avslice}&apikey={key}")
    for row in csv.reader(StringIO(response.content.decode())):
        if row[0] == "time":
            continue
        date_time = parse(row[0])
        date_key = datetime_to_key(date_time)
        timed_dict = timed_data.get(date_key, {})
        stock_dict = timed_dict.get(STOCKS_KEY, {})
        stock_times = stock_dict.get(stock_ticker, [])
        stock_times.append({
            "time": time.mktime(date_time.timetuple()),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
        stock_times = sorted(stock_times, key=lambda x: x["time"])
        stock_dict[stock_ticker] = stock_times
        timed_dict[STOCKS_KEY] = stock_dict
        timed_data[date_key] = timed_dict
    return timed_data


class TimedDataIterator:
    """Iterator class for iterating across timed data loads."""
    def __init__(self, json_files):
        """Initialise the iterator."""
        self.json_files_iter = iter(json_files)


    def __next__(self):
        """Returns the next values from the timed database."""
        with open(next(self.json_files_iter)) as json_file_handle:
            return json.load(json_file_handle)


class TimedDataLoader:
    """The class for storing timed data."""

    def __init__(self, alphavantage_key, data_folder):
        """Initialise the timed data class."""
        self.alphavantage_key = alphavantage_key
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        self.data_folder = data_folder
        self.json_files = sorted(glob.glob(os.path.join(data_folder, "*.json")))


    def extract(self):
        """Extract the training information."""
        timed_data = {}
        # Mine data from alphavantage
        for year in range(2):
            for month in range(12):
                for stock_ticker in STOCK_TICKERS:
                    avslice = f"year{year + 1}month{month + 1}"
                    print(f"Loading AlphaVantage {stock_ticker} intraday extended time series {avslice}")
                    timed_data = load_alphavantage_time_series_intraday_extended(stock_ticker, avslice, timed_data, self.alphavantage_key)
        # Write to JSON files
        for timed_data_key in timed_data:
            with open(os.path.join(self.data_folder, f"{timed_data_key}.json"), "w") as json_file_handle:
                json.dump(timed_data[timed_data_key], json_file_handle)


    def has_data(self):
        """Whether the data loader currently has data."""
        return bool(self.json_files)


    def __iter__(self):
        """Generate an iterator for the database."""
        return TimedDataIterator(self.json_files)


    def random(self):
        """Randomly access an item in the database."""
        with open(random.choice(self.json_files)) as json_file_handle:
            return json.load(json_file_handle)
