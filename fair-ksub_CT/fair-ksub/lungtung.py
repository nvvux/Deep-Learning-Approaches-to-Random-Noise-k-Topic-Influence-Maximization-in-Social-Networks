import pandas as pd
import numpy as np

df = pd.read_csv('graph1000.csv')
print(df.head())

S = [{457, 364, 863, 914, 573}, {311, 997, 890, 611, 883}]
print("Union:", set().union(*S))
print("Union size:", len(set().union(*S)))
spread = []
for _ in range(100):
    A = set([1, 2, 3])
    new_active = set([1, 2, 3])
    for i in range(10):
        new_ones = set([4, 5])
        new_active = new_ones - A
        A |= new_active
    spread.append(len(A))
print("Spread:", np.mean(spread))
