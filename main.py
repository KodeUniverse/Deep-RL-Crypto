import gym
import json
import datetime as dt
import pandas as pd
import os
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


from env.StockTradingEnv import StockTradingEnv

import pandas as pd

# df = pd.read_csv('./data/AAPL.csv')
# df = df.sort_values('Date')
BTC = pd.read_csv("data/CBSE_BTC_clean1.csv")
BTC = BTC.loc[:, "timestamp": "volume"]
BTC = BTC.rename(columns={"open":"Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

df = BTC

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
#model = DQN(MlpPolicy, env, verbose=1)
#model.learn(total_timesteps=20000)
model.learn(total_timesteps=1000)


if(os.path.exists("output/networth.csv")):
    os.remove("output/networth.csv")    

obs = env.reset()
for i in range(3000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()


## Graphing Net Worth

networth_arr = pd.read_csv("output/networth.csv")
plt.plot(networth_arr, color='firebrick')
plt.ticklabel_format(style='plain')
plt.title("Net Worth")
plt.savefig("output/graph.png")