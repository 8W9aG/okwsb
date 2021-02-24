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


STOCKS_KEY = "stocks"
TRAINING_DATA_FOLDER = ".training_data"
OPEN_KEY = "open"
HIGH_KEY = "high"
LOW_KEY = "low"
CLOSE_KEY = "close"
VOLUME_KEY = "volume"


def datetime_to_key(datetime):
    """Convert a datetime to a key."""
    return datetime.strftime("%Y_%m_%d")


def load_alphavantage_time_series_intraday_extended(stock_ticker, avslice, timed_data, key):
    """Load the alphavantage time series intraday extended data."""
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={stock_ticker}&interval=1min&slice={avslice}&apikey={key}")
    for row in csv.reader(StringIO(response.content.decode())):
        if row[0] == "time" or row[0] == "{":
            continue
        date_time = parse(row[0])
        date_key = datetime_to_key(date_time)
        timed_dict = timed_data.get(date_key, {})
        stock_dict = timed_dict.get(STOCKS_KEY, {})
        stock_times = stock_dict.get(stock_ticker, [])
        stock_times.append({
            "time": time.mktime(date_time.timetuple()),
            OPEN_KEY: float(row[1]),
            HIGH_KEY: float(row[2]),
            LOW_KEY: float(row[3]),
            CLOSE_KEY: float(row[4]),
            VOLUME_KEY: float(row[5]),
        })
        stock_times = sorted(stock_times, key=lambda x: x["time"])
        stock_dict[stock_ticker] = stock_times
        timed_dict[STOCKS_KEY] = stock_dict
        timed_data[date_key] = timed_dict
    return timed_data


def load_alphavantage_stock_tickers(key):
    """Load the alphavantage stock ticker list."""
    response = requests.get(f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={key}")
    stock_tickers = []
    for row in csv.reader(StringIO(response.content.decode())):
        if row[0] == "symbol":
            continue
        stock_tickers.append(row[0])
    return stock_tickers


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

    def __init__(self, alphavantage_key, data_folder, stock_tickers_max=8, stock_tickers=None):
        """Initialise the timed data class."""
        self.alphavantage_key = alphavantage_key
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)
        self.data_folder = data_folder
        self.json_files = sorted(glob.glob(os.path.join(data_folder, "*.json")))
        self.stock_tickers_max = stock_tickers_max
        self.input_stock_tickers = [] if stock_tickers is None else stock_tickers


    def extract(self):
        """Extract the training information."""
        timed_data = {}
        # Mine data from alphavantage
        stock_tickers = load_alphavantage_stock_tickers(self.alphavantage_key)
        download_stock_tickers = [x for x in stock_tickers if x in self.input_stock_tickers]
        download_stock_tickers.extend([x for x in stock_tickers if x not in self.input_stock_tickers][:self.stock_tickers_max])
        for year in range(2):
            for month in range(12):
                for stock_ticker in download_stock_tickers:
                    avslice = f"year{year + 1}month{month + 1}"
                    print(f"Loading AlphaVantage {stock_ticker} intraday extended time series {avslice}")
                    timed_data = load_alphavantage_time_series_intraday_extended(stock_ticker, avslice, timed_data, self.alphavantage_key)
                    time.sleep(20)
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


    def stock_tickers(self):
        """Find the amount of stock tickers in the database."""
        tickers = set()
        for json_file in self.json_files:
            with open(json_file) as json_file_handle:
                timed_data = json.load(json_file_handle)
                for ticker in timed_data[STOCKS_KEY]:
                    tickers.add(ticker)
        return list(tickers)
