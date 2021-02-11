"""The main runner."""
import gym

from .stockenv import ENVIRONMENT_ID


def main() -> None:
    """Run OKWSB."""
    gym.envs.registration.register(id=ENVIRONMENT_ID, entry_point='okwsb:StockEnv') 
    env = gym.make(ENVIRONMENT_ID)
    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
        print(f"Stepping")
    env.close()


if __name__ == "__main__":
    main()  # pragma: no cover