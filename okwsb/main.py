"""The main runner."""
import gym
from stable_baselines.common.env_checker import check_env

from .stockenv import ENVIRONMENT_ID


def main() -> None:
    """Run OKWSB."""
    print("--- OKWSB ---")
    gym.envs.registration.register(id=ENVIRONMENT_ID, entry_point='okwsb:StockEnv') 
    env = gym.make(ENVIRONMENT_ID, capital = 100000)
    check_env(env)
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        print(f"Stepping")
    env.close()


if __name__ == "__main__":
    main()  # pragma: no cover
