"""The main runner."""
import argparse
import time
import sys

import gym
from stable_baselines.common.env_checker import check_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import tensorflow as tf

from .stockenv import ENVIRONMENT_ID, MAXIMUM_STOCK_PRICE
from .timed_data import TimedDataLoader, TRAINING_DATA_FOLDER


MODE_DATA = "data"
MODE_TRAIN = "train"
MODE_TEST = "test"
MODE_LIVE = "live"

def main() -> None:
    """Run OKWSB."""
    print("--- OKWSB ---")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--starting_capital', type=int, default=100000, help="How much capital is the training starting with")
    parser.add_argument('--training_timesteps', type=int, default=1000000, help="How many timesteps are in the training")
    parser.add_argument("--model_name", default="okwsb-pp02", help="The name of the model file to write")
    parser.add_argument('--mode', choices=(MODE_DATA, MODE_TRAIN, MODE_TEST, MODE_LIVE), required=True, help="The mode to run OKWSB in")
    parser.add_argument("--alphavantage_key", required=True, help="The API key for interfacing to AlphaVantage")
    parser.add_argument("--data_folder", default=TRAINING_DATA_FOLDER, help="The folder to store the training data in")
    parser.add_argument("--data_stock_tickers_max", default=16, help="The maximum amount of stock tickers")
    parser.add_argument("--data_stock_tickers", required=False, nargs="+", help="The stock tickers to download")
    args = parser.parse_args()
    # Consolidate data
    timed_data = TimedDataLoader(
        args.alphavantage_key,
        args.data_folder,
        stock_tickers_max=args.data_stock_tickers_max,
        stock_tickers=args.data_stock_tickers)
    if not timed_data.has_data() and args.mode != MODE_DATA:
        print("No data found. Try `okwsb --mode=data` to collect a local dataset.")
        sys.exit(1)
    if args.mode == MODE_DATA:
        timed_data.extract()
    else:
        # Validate environment
        gym.envs.registration.register(id=ENVIRONMENT_ID, entry_point='okwsb:StockEnv') 
        env = gym.make(
            ENVIRONMENT_ID,
            capital = args.starting_capital,
            timed_data = timed_data,
            playback = args.mode == MODE_TEST)
        check_env(env)
        env.reset()
        if args.mode == MODE_TRAIN:
            model = PPO2(MlpPolicy, env, verbose = 1)
            model.learn(total_timesteps = args.training_timesteps)
            model.save(args.model_name)
            env.reset()
        elif args.mode == MODE_TEST:
            if not timed_data.has_data():
                timed_data.extract()
            model = PPO2.load(args.model_name)
            obs = env.reset()
            try:
                while True:
                    action, _ = model.predict(obs)
                    obs, _, done, _ = env.step(action)
                    if done:
                        env.render()
                        obs = env.reset()
            except StopIteration:
                pass
        elif args.mode == MODE_LIVE:
            pass
        env.close()


if __name__ == "__main__":
    main()  # pragma: no cover
