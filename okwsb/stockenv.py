"""The stock trading environment class."""
import gym
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from gym import spaces


ENVIRONMENT_ID = "stockenv-v0"
HUMAN_RENDER_MODE = "human"
RGBARRAY_RENDER_MODE = "rgb_array"
MAXIMUM_STOCK_PRICE = 1000000


class StockEnv(gym.Env):
    """The environment representing a stock trading world."""

    def __init__(self, capital = 100000):
        """Initialise the stock environment."""
        super(StockEnv, self).__init__()
        self.current_usd = capital
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, MAXIMUM_STOCK_PRICE]), dtype=np.float16)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([MAXIMUM_STOCK_PRICE]), dtype=np.float16)
        self.reset()


    def step(self, action):
        """Take an action in the environment and advance to the next state."""
        reward = 0
        action_type = round(action[0])
        if action_type == 2:
            reward = 1
        else:
            reward = -1
        done = True
        return self.state, reward, done, {}


    def reset(self):
        """Reset the state of the environment."""
        self.state = np.array([0])
        return self.state


    def render(self, mode = HUMAN_RENDER_MODE, close = False):
        """Render the current state."""
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3])
        canvas.draw()
        buf = canvas.buffer_rgba()
        x = np.asarray(buf) / 255.0
        if mode == HUMAN_RENDER_MODE:
            cv2.imshow("Stocks", x)
            cv2.waitKey(1)
        elif mode == RGBARRAY_RENDER_MODE:
            return x
