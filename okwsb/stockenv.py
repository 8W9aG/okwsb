"""The stock trading environment class."""
import math
import enum
from datetime import datetime

import gym
import numpy as np
from gym import spaces

from .timed_data import CLOSE_KEY, HIGH_KEY, LOW_KEY, STOCKS_KEY, OPEN_KEY, VOLUME_KEY


class StockAction(enum.IntEnum):
    """Actions to perform on the stock."""
    Hold = 0
    Buy = 1
    Sell = 2


ENVIRONMENT_ID = "stockenv-v0"
HUMAN_RENDER_MODE = "human"
RGBARRAY_RENDER_MODE = "rgb_array"
MAXIMUM_STOCK_PRICE = 1000000.0
# Actions
ACTIONS_PER_STOCK = len(StockAction)
# Observations
STATIC_OBSERVATIONS = 1
OBSERVATIONS_PER_STOCK = 6
STOCK_OBSERVATIONS_ROWS = 2
STOCK_POSITION_INDEX = 0
STOCK_OPEN_INDEX = 1
STOCK_CLOSE_INDEX = 2
STOCK_HIGH_INDEX = 3
STOCK_LOW_INDEX = 4
STOCK_VOLUME_INDEX = 5


def normalise(number, min_range = 0.0, max_range = MAXIMUM_STOCK_PRICE, min_range_normalised = -1.0, max_range_normalised = 1.0):
    """Normalise a number to within a minimum and maximum."""
    clipped_number = min(max(number, min_range), max_range)
    normalised_number = clipped_number / abs(max_range - min_range)
    normal_range = abs(max_range_normalised - min_range_normalised)
    restretched_number = normalised_number * normal_range
    return restretched_number - (normal_range / 2.0)


def denormalise(number, min_range = 0.0, max_range = MAXIMUM_STOCK_PRICE, min_range_normalised = -1.0, max_range_normalised = 1.0):
    """Denormalise a number to within a minimum and a maximum."""
    clipped_number = min(max(number, min_range_normalised), max_range_normalised)
    normal_range = abs(max_range_normalised - min_range_normalised)
    positive_number = clipped_number + (normal_range / 2.0)
    normalised_number = positive_number / normal_range
    return normalised_number * max_range


class StockEnv(gym.Env):
    """The environment representing a stock trading world."""

    def __init__(self, capital = 100000, timed_data = None, playback = False):
        """Initialise the stock environment."""
        super(StockEnv, self).__init__()
        self._starting_usd = capital
        self._stock_tickers = timed_data.stock_tickers()
        self._last_nonzero_stock_prices = {x : 0.0 for x in self._stock_tickers}
        self.action_space = spaces.Box(
            low=np.ones(len(self._stock_tickers) * ACTIONS_PER_STOCK) * -1.0,
            high=np.ones(len(self._stock_tickers) * ACTIONS_PER_STOCK),
        )
        self.observation_space = spaces.Box(
            low=np.ones((len(self._stock_tickers) * OBSERVATIONS_PER_STOCK * STOCK_OBSERVATIONS_ROWS) + STATIC_OBSERVATIONS) * -1.0,
            high=np.ones((len(self._stock_tickers) * OBSERVATIONS_PER_STOCK * STOCK_OBSERVATIONS_ROWS) + STATIC_OBSERVATIONS),
        )
        self._timed_data = timed_data
        self._timed_data_iterator = iter(timed_data)
        self._playback = playback
        self._current_usd = self._starting_usd
        self._positions = [0 for _ in range(len(self._stock_tickers))]
        self.reset()


    def timed_data_to_state(self):
        """Convert timed data to a state."""
        state = np.zeros(STATIC_OBSERVATIONS + (OBSERVATIONS_PER_STOCK * len(self._stock_tickers)))
        state[0] = normalise(self._current_usd)
        for i in range(len(self._stock_tickers)):
            state_dict = {
                OPEN_KEY: 0.0,
                CLOSE_KEY: 0.0,
                HIGH_KEY: 0.0,
                LOW_KEY: 0.0,
                VOLUME_KEY: 0.0
            }
            stock_key = self._stock_tickers[i]
            if stock_key in self._current_timed_data[STOCKS_KEY]:
                stock_list = self._current_timed_data[STOCKS_KEY][stock_key]
                if len(stock_list) > self._step:
                    state_dict = stock_list[self._step]
            for j in range(STOCK_OBSERVATIONS_ROWS - 1):
                stock_start_index = (j * OBSERVATIONS_PER_STOCK * len(self._stock_tickers)) + (i * OBSERVATIONS_PER_STOCK) + STATIC_OBSERVATIONS
                stock_next_index = ((j + 1) * OBSERVATIONS_PER_STOCK * len(self._stock_tickers)) + (i * OBSERVATIONS_PER_STOCK) + STATIC_OBSERVATIONS
                for k in range(OBSERVATIONS_PER_STOCK):
                    state[stock_next_index + k] = state[stock_start_index + k]
            stock_start_index = (i * OBSERVATIONS_PER_STOCK) + STATIC_OBSERVATIONS
            state[stock_start_index + STOCK_POSITION_INDEX] = normalise(self._positions[i])
            state[stock_start_index + STOCK_OPEN_INDEX] = normalise(state_dict[OPEN_KEY])
            state[stock_start_index + STOCK_CLOSE_INDEX] = normalise(state_dict[CLOSE_KEY])
            state[stock_start_index + STOCK_HIGH_INDEX] = normalise(state_dict[HIGH_KEY])
            state[stock_start_index + STOCK_LOW_INDEX] = normalise(state_dict[LOW_KEY])
            state[stock_start_index + STOCK_VOLUME_INDEX] = normalise(state_dict[VOLUME_KEY])
            if state_dict[CLOSE_KEY] > 0.0:
                self._last_nonzero_stock_prices[stock_key] = state_dict[CLOSE_KEY]
        return state


    def stock_price(self, stock_index):
        """Find the stock price for a specific stock."""
        return denormalise(self.state[(stock_index * OBSERVATIONS_PER_STOCK) + STOCK_OPEN_INDEX + STATIC_OBSERVATIONS])


    def calculate_delta_value_usd(self):
        """Calculate the delta USD value of the positions."""
        stock_value = sum([self.stock_price(i) * self._positions[i] for i in range(len(self._stock_tickers))])
        return self._current_usd + stock_value - self._starting_usd


    def step(self, action):
        """Take an action in the environment and advance to the next state."""
        for i in range(len(self._stock_tickers)):
            action_type = min(round(((action[(i * ACTIONS_PER_STOCK)] + 1.0) / 2.0) * ACTIONS_PER_STOCK), ACTIONS_PER_STOCK - 1)
            action_number = round(((action[(i * ACTIONS_PER_STOCK) + 1] + 1.0) / 2.0) * MAXIMUM_STOCK_PRICE)
            stock_price = self.stock_price(i)
            # Force a sale if the stock is taken off the market
            if stock_price <= 0.0 and self._positions[i] > 0:
                action_type = StockAction.Sell
                action_number = MAXIMUM_STOCK_PRICE
                stock_price = self._last_nonzero_stock_prices[self._stock_tickers[i]]
            if action_type == StockAction.Hold:
                pass
            elif action_type == StockAction.Buy:
                if stock_price > 0.0:
                    stocks = int(min(action_number, math.floor(self._current_usd / stock_price)))
                    if stocks > 0:
                        print(f"Buy {self._stock_tickers[i]} {stocks}")
                        self._current_usd -= stocks * stock_price
                        self._positions[i] += stocks
            elif action_type == StockAction.Sell:
                stocks = min(action_number, self._positions[i])
                if stocks > 0:
                    print(f"Sell {self._stock_tickers[i]} {stocks}")
                    self._current_usd += stocks * stock_price
                    self._positions[i] -= stocks
        self._step += 1
        self.state = self.timed_data_to_state()
        reward = self.calculate_delta_value_usd()
        return self.state, reward, self._step >= (self._max_steps - 1), {}


    def reset(self):
        """Reset the state of the environment."""
        self._current_timed_data = self._timed_data.random() if not self._playback else next(self._timed_data_iterator)
        self._step = 0
        self._max_steps = len(self._current_timed_data[STOCKS_KEY][list(self._current_timed_data[STOCKS_KEY].keys())[0]])
        for ticker in self._current_timed_data[STOCKS_KEY]:
            self._max_steps = min(len(self._current_timed_data[STOCKS_KEY][ticker]), self._max_steps)
        if not self._playback:
            self._current_usd = self._starting_usd
            self._positions = [0 for _ in range(len(self._stock_tickers))]
        self.state = self.timed_data_to_state()
        return self.state


    def render(self, mode = HUMAN_RENDER_MODE, close = False):
        """Render the current state."""
        stock_tickers = list(self._current_timed_data[STOCKS_KEY].keys())
        datetime_string = datetime.fromtimestamp(self._current_timed_data[STOCKS_KEY][stock_tickers[0]][0]["time"]).strftime("%Y-%m-%d")
        print(f"--- {datetime_string}")
        print(f"Current USD: ${self._current_usd}")
        for i in range(len(self._stock_tickers)):
            if self._positions[i] == 0:
                continue
            stock_price = self.stock_price(i)
            print(f"{self._stock_tickers[i]}\t{self._positions[i]}\t${stock_price}\t${stock_price * self._positions[i]}")
        print(f"Delta USD: ${self.calculate_delta_value_usd()}")
        print("---")
