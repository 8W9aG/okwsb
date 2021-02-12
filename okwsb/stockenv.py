"""The stock trading environment class."""
import itertools
import math

import gym
import numpy as np
from gym import spaces

from .timed_data import STOCKS_KEY


ENVIRONMENT_ID = "stockenv-v0"
HUMAN_RENDER_MODE = "human"
RGBARRAY_RENDER_MODE = "rgb_array"
MAXIMUM_STOCK_PRICE = 1000000.0
# Actions
ACTION_TYPE_HOLD = 0
ACTION_TYPE_BUY = 1
ACTION_TYPE_SELL = 2
ACTIONS_PER_STOCK = 2
# Observations
OBSERVATIONS_PER_STOCK = 2
STOCK_POSITION_INDEX = 0
STOCK_OPEN_INDEX = 1


def timed_data_to_state(timed_data, step, positions):
    """Convert timed data to a state."""
    stock_keys = list(timed_data[STOCKS_KEY].keys())
    #print(f"{step}")
    return np.array(list(itertools.chain(*[[
        positions[count] / MAXIMUM_STOCK_PRICE,
        timed_data[STOCKS_KEY][ticker][step]["open"] / MAXIMUM_STOCK_PRICE,
    ] for count, ticker in enumerate(stock_keys)])))


class StockEnv(gym.Env):
    """The environment representing a stock trading world."""

    def __init__(self, capital = 100000, timed_data = None):
        """Initialise the stock environment."""
        super(StockEnv, self).__init__()
        self._starting_usd = capital
        example_timed_data = timed_data.random()
        self._number_stocks = len(example_timed_data[STOCKS_KEY])
        self.action_space = spaces.Box(
            low=np.array([-1 for _ in range(self._number_stocks * ACTIONS_PER_STOCK)]),
            high=np.array([1 for _ in range(self._number_stocks * ACTIONS_PER_STOCK)]))
        self.observation_space = spaces.Box(
            low=np.array([-1 for _ in range(self._number_stocks * OBSERVATIONS_PER_STOCK)]), # ADD IN CURRENTUSD
            high=np.array([1 for _ in range(self._number_stocks * OBSERVATIONS_PER_STOCK)]))
        self._timed_data = timed_data
        self.reset()


    def stock_price(self, stock_index):
        """Find the stock price for a specific stock."""
        return self.state[(stock_index * OBSERVATIONS_PER_STOCK) + STOCK_OPEN_INDEX] * MAXIMUM_STOCK_PRICE


    def calculate_delta_value_usd(self):
        """Calculate the delta USD value of the positions."""
        stock_value = sum([self.stock_price(i) * self._positions[i] for i in range(self._number_stocks)])
        return self._current_usd + stock_value - self._starting_usd


    def step(self, action):
        """Take an action in the environment and advance to the next state."""
        for i in range(self._number_stocks):
            action_type = round(((action[(i * ACTIONS_PER_STOCK)] + 1.0) / 2.0) * 3.0) # Fix this 3.0 value
            action_number = round(((action[(i * ACTIONS_PER_STOCK) + 1] + 1.0) / 2.0) * MAXIMUM_STOCK_PRICE)
            if action_type == ACTION_TYPE_HOLD:
                pass
            elif action_type == ACTION_TYPE_BUY:
                stock_price = self.stock_price(i)
                stocks = int(min(action_number, math.floor(self._current_usd / stock_price)))
                self._current_usd -= stocks * stock_price
                self._positions[i] += stocks
            elif action_type == ACTION_TYPE_SELL:
                stock_price = self.stock_price(i)
                stocks = min(action_number, self._positions[i])
                self._current_usd += stocks * stock_price
                self._positions[i] -= stocks
        self._step += 1
        self.state = timed_data_to_state(self._current_timed_data, self._step, self._positions)
        return self.state, self.calculate_delta_value_usd(), self._step >= (self._max_steps - 1), {}


    def reset(self):
        """Reset the state of the environment."""
        self._current_timed_data = self._timed_data.random()
        self._step = 0
        self._max_steps = len(self._current_timed_data[STOCKS_KEY][list(self._current_timed_data[STOCKS_KEY].keys())[0]])
        for ticker in self._current_timed_data[STOCKS_KEY]:
            self._max_steps = min(len(self._current_timed_data[STOCKS_KEY][ticker]), self._max_steps)
        self._current_usd = self._starting_usd
        self._positions = [0 for _ in range(self._number_stocks)]
        self.state = timed_data_to_state(self._current_timed_data, self._step, self._positions)
        return self.state


    def render(self, mode = HUMAN_RENDER_MODE, close = False):
        """Render the current state."""
        print("---")
        print(f"Current USD: ${self._current_usd}")
        stock_tickers = list(self._current_timed_data[STOCKS_KEY].keys())
        for i in range(self._number_stocks):
            if self._positions[i] == 0:
                continue
            stock_price = self.stock_price(i)
            print(f"Stock {stock_tickers[i]}: {self._positions[i]} ${stock_price} ${stock_price * self._positions[i]}")
        print(f"Delta USD: ${self.calculate_delta_value_usd()}")
        print("---")
