import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns

df_untied = pd.read_csv('./not_tied.csv')
df_tied = pd.read_csv('./tied.csv')

iterations = [1,2,5,10,20,50]
cluster_nums = [1,2,3,5,7,10,20,50,100]

xs_untied = df_untied['cluster_num'].values.tolist()
ys_untied = df_untied['iterations'].values.tolist()
train_LL_untied = df_untied['train_LL'].values.tolist()
dev_LL_untied = df_untied['dev_LL'].values.tolist()

xs_tied = df_tied['cluster_num'].values.tolist()
ys_tied = df_tied['iterations'].values.tolist()
train_LL_tied = df_tied['train_LL'].values.tolist()
dev_LL_tied = df_tied['dev_LL'].values.tolist()

f, axarr = plt.subplots(2, sharex=True)
f2, axarr2 = plt.subplots(2, sharex=True)


sns.set()
pal = sns.color_palette("Set3", 9)
pal2 = sns.color_palette("Set3", 9)

train_ll_untied = list()
train_ll_tied = list()
dev_ll_untied = list()
dev_ll_tied = list()


for i in range(len(cluster_nums)):
    train_ll_untied.append(df_untied[df_untied.cluster_num == cluster_nums[i]]['train_LL'].tolist())
    axarr[0].plot(iterations, train_ll_untied[i], c=pal[i], label='cluster_num={}'.format(cluster_nums[i]))
    axarr[0].legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    axarr[0].set_ylim(-4.9,-4)
    axarr[0].set_title('Training')

    dev_ll_untied.append(df_untied[df_untied.cluster_num == cluster_nums[i]]['dev_LL'].tolist())
    axarr[1].plot(iterations, dev_ll_untied[i], c=pal[i], label='cluster_num={}'.format(cluster_nums[i]))
    axarr[1].set_title('Dev')

    train_ll_tied.append(df_tied[df_tied.cluster_num == cluster_nums[i]]['train_LL'].tolist())
    axarr2[0].plot(iterations, train_ll_tied[i], c=pal[i], label='cluster_num={}'.format(cluster_nums[i]))
    axarr2[0].legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    axarr2[0].set_ylim(-4.9,-4)
    axarr2[0].set_title('Training')

    dev_ll_tied.append(df_tied[df_tied.cluster_num == cluster_nums[i]]['dev_LL'].tolist())
    axarr2[1].plot(iterations, dev_ll_tied[i], c=pal[i], label='cluster_num={}'.format(cluster_nums[i]))
    axarr2[1].set_title('Dev')





# fig = plt.figure()
# ax = plt.axes(projection='3d')

# xs = df_untied['cluster_num'].values.tolist()
# ys = df_untied['iterations'].values.tolist()
# train_LL = df_untied['train_LL'].values.tolist()
# dev_LL = df_untied['dev_LL'].values.tolist()

# for x, y, z in zip(xs, ys, train_LL):
#     label = '(%d, %d)' % (x, y)
#     ax.text(x, y, z, label)


# ax.set_xlabel('cluster number')
# ax.set_ylabel('iterations')
# ax.set_zlabel('train log likelihood')
# ax.scatter3D(xs, ys, train_LL, c=train_LL, label='not tied graph')

f.savefig('graph1.png', dpi=300)
f2.savefig('graph2.png', dpi=300)