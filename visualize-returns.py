import matplotlib.pyplot as plt
import pandas as pd
import os

LSTM1 = pd.read_csv("agents/LSTM1/networth.csv")
LSTM2 = pd.read_csv("agents/LSTM2/networth.csv")
MLP1 = pd.read_csv("agents/MLP1/networth.csv")
MLP2 = pd.read_csv("agents/MLP2/networth.csv")


df = pd.concat((LSTM1, LSTM2, MLP1, MLP2), axis=1)
plt.plot(df)
plt.ticklabel_format(style='plain')
plt.title("RL Agents Net Worth")
plt.legend(("LSTM1", "LSTM2", "MLP1", "MLP2"))
plt.savefig("output/RL-agents.png")

def simple_return(x):
    return (x.iloc[-1] / x.iloc[0] - 1)

print(f"LSTM1 return: {simple_return(LSTM1)[0]*100:.2f}%")
print(f"LSTM2 return: {simple_return(LSTM2)[0]*100:.2f}%")
print(f"MLP1 return: {simple_return(MLP1)[0]*100:.2f}%")
print(f"MLP2 return: {simple_return(MLP2)[0]*100:.2f}%")
