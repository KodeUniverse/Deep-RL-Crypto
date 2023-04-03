import gym
import json
import datetime as dt
import pandas as pd
import os
import matplotlib.pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


from env.StockTradingEnv import StockTradingEnv


BTC = pd.read_csv("data/CBSE_BTC_clean1.csv")
BTC = BTC.loc[:, "timestamp": "volume"]
BTC = BTC.rename(columns={"open":"Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})


slice_point = round(len(BTC)* .70)
traindata = BTC[:slice_point]
testdata = BTC[slice_point:].reset_index()


# The algorithms require a vectorized environment to run
train_env = DummyVecEnv([lambda: StockTradingEnv(traindata)])
test_env = DummyVecEnv([lambda: StockTradingEnv(testdata)])

model = PPO2(MlpPolicy, train_env, verbose=1, nminibatches=1)

model.learn(total_timesteps=200_000)


if(os.path.exists("output/networth.csv")):
    os.remove("output/networth.csv")    

obs = test_env.reset()
for i in range(200_000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = test_env.step(action)
    test_env.render()


## Graphing Net Worth

networth_arr = pd.read_csv("output/networth.csv")
plt.plot(networth_arr, color='firebrick')
plt.ticklabel_format(style='plain')
plt.title("Net Worth")
plt.savefig("output/graph.png")