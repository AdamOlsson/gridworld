from gridworldenv import GridWorldEnv
import numpy as np

env = GridWorldEnv()

for i in range(16):
    print(env.P[i][3])