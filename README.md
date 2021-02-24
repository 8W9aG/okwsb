# okwsb

A stock trading bot based on Reinforcement Learning.

## Raison D'être :thought_balloon:

If machines can learn to play Super Smash Brothers, why can they not learn how to trade stocks? This repository tries to apply the discipline of Reinforcement Learning to the problem of trading stocks on an intraday basis with the goal of achieving maximum profit.

## Architecture :triangular_ruler:

`okwsb` has 4 modes.

The modes of `okwsb` are as follows:

- **Data** Collect training data from free information sources.
- **Train** Train the RL algorithm on the collected data by showing it random days.
- **Test** Test the RL algorithm on the collected data in order and score it based on the profit.
- **Live** Run the RL algorithm on a live trading account instance (can be done dry or with human intervention).

## Dependencies :globe_with_meridians:

- [OpenAI gym](https://gym.openai.com/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/stable/index.html)
- [OpenCV](https://opencv.org/)
- [Pillow](https://python-pillow.org/)
- [Stable-Baselines](https://stable-baselines3.readthedocs.io/en/master/)
- [Tensorflow](https://www.tensorflow.org/)
- [AlphaVantage](https://www.alphavantage.co/)
- [Requests](https://2.python-requests.org/en/master/)
- [python-dateutil](https://pypi.org/project/python-dateutil/)

## Installation :inbox_tray:

`okwsb` is a [python](https://www.python.org/) project, to install and run it is first recommended to add a virtual environment like so:

```shell
$ python3 -m venv venv
$ source venv/bin/activate
```

Then perform the installation:

```shell
$ python setup.py install
```

## Usage example :eyes:

To collect data for the system run the following:

```shell
$ okwsb --mode=data --alphavantage_key=YOUR_KEY_HERE
```

To train the system run the following:

```shell
$ okwsb --mode=train --alphavantage_key=YOUR_KEY_HERE
```

To test the system run the following:

```shell
$ okwsb --mode=test
```

## License :memo:

The project is available under the MIT license.
