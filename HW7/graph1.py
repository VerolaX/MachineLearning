import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

df_untied = pd.read_csv('./not_tied.csv')
fig = plt.figure()
ax = plt.axes(projection='3d')

xs = df_untied['cluster_num'].values.tolist()
ys = df_untied['iterations'].values.tolist()
train_LL = df_untied['train_LL'].values.tolist()
dev_LL = df_untied['dev_LL'].values.tolist()

for x, y, z in zip(xs, ys, train_LL):
    label = '(%d, %d)' % (x, y)
    ax.text(x, y, z, label)


ax.set_xlabel('cluster number')
ax.set_ylabel('iterations')
ax.set_zlabel('train log likelihood')
ax.scatter3D(xs, ys, train_LL, c=train_LL, label='not tied graph')

plt.savefig('graph1.png')