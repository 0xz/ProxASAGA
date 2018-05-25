
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


df1 = pd.read_csv('prox_to_graph', names=['Threads', 'Epoch', 'Time', 'Objective'])
df2 = pd.read_csv('tick_to_graph', names=['Threads', 'Epoch', 'Time', 'Objective'])

#millise = lambda x: float(x) * 1000
#df1['Time'] = df1['Time'].apply(millise)

ls = [':', '-.', '--']

fig, ax_list = plt.subplots(1, 2, figsize=(10, 5))

for i, t in enumerate([ 2, 4, 6, 8, 10, 12, 14, 16]):
  print("t ", t)

  df1_fit = df1[(df1['Threads'] == t)]
  df2_fit = df2[(df2['Threads'] == t)]

  prox_obj_array = df1_fit['Objective']
  prox_min_obj = prox_obj_array.min()

  tick_obj_array = df2_fit['Objective']
  tick_min_obj = tick_obj_array.min()

  real_min=min([prox_min_obj, tick_min_obj])

  ax_list[0].plot(df1_fit["Time"], prox_obj_array - real_min, label="prox " + str(t), c='C'+str(i))
  ax_list[0].plot(df2_fit["Time"], tick_obj_array - real_min, label="tick " + str(t), linestyle='--', c='C'+str(i))
  ax_list[0].set_xlabel("time")

  ax_list[1].plot(df1_fit["Epoch"], prox_obj_array - real_min, label="prox " + str(t), c='C'+str(i))
  ax_list[1].plot(df2_fit["Epoch"], tick_obj_array - real_min, label="tick " + str(t), linestyle='--', c='C'+str(i))
  ax_list[1].set_xlabel("epochs")

ax_list[0].set_yscale("log")
ax_list[1].set_yscale("log")

ax_list[0].legend()
ax_list[1].legend()

plt.savefig("newplots.png")
