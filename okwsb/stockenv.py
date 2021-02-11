"""The stock trading environment class."""
import gym
import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


ENVIRONMENT_ID = "stockenv-v0"
HUMAN_RENDER_MODE = "human"
RGBARRAY_RENDER_MODE = "rgb_array"


class StockEnv(gym.Env):
    """The environment representing a stock trading world."""

    def __init__(self):
        """Initialise the stock environment."""
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Discrete(2)
        self.state = []


    def step(self, action):
        """Take an action in the environment and advance to the next state."""
        reward = 0
        if action == 2:
            reward = 1
        else:
            reward = -1
        done = True
        return self.state, reward, done, {}


    def reset(self):
        """Reset the state of the environment."""
        self.state = []
        return self.state


    def render(self, mode = HUMAN_RENDER_MODE, close = False):
        """Render the current state."""
        fig = Figure(figsize=(5,4), dpi=100)
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
