import gymnasium as gym
from gymnasium import spaces
import random
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import random
from stable_baselines3.common.utils import get_schedule_fn
import os
from env import StockTradingEnv, LSTMv1


### tuning parameters for continuous learning
def tune_parameters(model, ep):
    model.learning_rate = get_schedule_fn(3e-5 * 0.95)
    model.ent_coef = 0.01
    model.clip_range = get_schedule_fn(0.2)
    model.n_epochs = 5
    return model


## trading test
def trade(df: pd.DataFrame, episodes: int):
    # Load saved model
    models_dir = "C:/Users/RUTHVIK REDDY/StockSentinel/models/PPO"
    model_path = f"{models_dir}/14M_02.zip"

    # Initialize the environment
    env = DummyVecEnv([lambda: StockTradingEnv(df=df, render_mode="human")])
    env.seed(0)
    env.reset()

    # Load PPO model
    model = PPO.load(model_path, env=env)

    # Initialize a dictionary to store results
    results = {
        "sharpe_ratios": [],
        "sortino_ratios": [],
        "cumulative_returns": [],
        "max_earning_rates": [],
        "max_drawdowns": [],
        "average_profitabilities": [],
        "max_pullbacks": [],
        "average_profitabilities_per_trade": [],
        "net_worth_logs": [],
        "rewards": [],
    }

    # Run episodes
    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_rewards = []

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward[0])
            env.render()

        # continuous learning
        model = tune_parameters(model, ep)
        model.learn(total_timesteps=1200, reset_num_timesteps=False, progress_bar=False)
        # change timesteps if needed

        # saving model
        model.save(model_path)

        # Extract relevant information from the environment's info
        info_dict = info[0]  # Unwrap from DummyVecEnv

        # Store the metrics for the current episode
        results["sharpe_ratios"].append(info_dict.get("sharpe_ratio", 0.0))
        results["sortino_ratios"].append(info_dict.get("sortino_ratio", 0.0))
        results["cumulative_returns"].append(info_dict.get("cumulative_return", 0.0))
        results["max_earning_rates"].append(info_dict.get("max_earning_rate", 0.0))
        results["max_drawdowns"].append(info_dict.get("max_drawdown", 0.0))
        results["average_profitabilities"].append(info_dict.get("average_profitability", 0.0))
        results["max_pullbacks"].append(info_dict.get("max_pullback", 0.0))
        results["average_profitabilities_per_trade"].append(info_dict.get("average_profitability_per_trade", 0.0))
        results["net_worth_logs"].append(info_dict.get("net_worth_log", []))
        results["rewards"].append(episode_rewards)  # Store rewards for this episode

    # Return the collected results
    return (
        np.mean(results["cumulative_returns"]),
        np.mean(results["max_earning_rates"]),
        np.mean(results["max_pullbacks"]),
        np.mean(results["average_profitabilities_per_trade"]),
        np.mean(results["sharpe_ratios"]),
        np.mean(results["sortino_ratios"]),
        np.mean(results["max_drawdowns"]),
        np.mean(results["average_profitabilities"]),
        results["net_worth_logs"],
        results["rewards"]
    )


# if __name__ == "__main__":
#     df = pd.read_csv("C:/Users/RUTHVIK REDDY/StockSentinel/data/^NSEI_test_data.csv")
#     episodes = 1
#     print(trade(df, episodes))