# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:37:23 2020

@author: Aiken

For running tSNE analysis on MN, FSGS, and IgAN datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import time

filePath = 'pathTo/file.csv'
# load dataset into Pandas DataFrame
df = pd.read_csv(filePath)
nrow, ncol = df.shape
healthyCtrl = 21

# Standardizing the data
norm_data = StandardScaler().fit_transform(df)

time_start = time.time()

tSNE_2D_data = TSNE(n_components=2).fit_transform(norm_data)
tSNE_3D_data = TSNE(n_components=3).fit_transform(norm_data)

print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

group = []
for i in range(nrow):
    if i<healthyCtrl:
        group.append('Healthy')
    elif i<48:
        group.append('IgAN')
    elif i<73:
       group.append('FSGS')
    else:
       group.append('MN')

groupColumnData = pd.read_csv('umap_'+filePath)

targets = ['Healthy', 'IgAN','FSGS','MN']
colors = ['b', 'r','g','orange']

# for 2D ploting
fig0 = plt.figure(figsize = (8,8))
ax0 = fig0.add_subplot(1,1,1) 
ax0.set_xlabel('Component-1', fontsize = 15)
ax0.set_ylabel('Component-2', fontsize = 15)
ax0.set_title(('tSNE - 2D'), fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = groupColumnData['Group'] == target
    ax0.scatter(tSNE_2D_data[indicesToKeep, 0]
               , tSNE_2D_data[indicesToKeep, 1]
               , c = color
               , s = 50)
ax0.legend(targets)

# for 3D ploting
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('\nComponent-1', fontsize = 15)
ax.set_ylabel('\nComponent-2', fontsize = 15)
ax.set_zlabel('\nComponent-3', fontsize = 15)
ax.tick_params(axis='x', which='major', pad=0.01)
plt.rc('xtick',labelsize=14)


ax.tick_params(axis='y', which='major', pad=0.01)
ax.tick_params(axis='z', which='major', pad=0.01)
ax.zaxis.labelpad = -5
ax.set_title(('tSNE - 3D'), fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = groupColumnData['Group'] == target
    ax.scatter(tSNE_3D_data[indicesToKeep, 0]
               , tSNE_3D_data[indicesToKeep, 1]
               , tSNE_3D_data[indicesToKeep, 2]
               , c = color
               , s = 50
               , depthshade = False)
ax.legend(targets)
ax.grid()
ax.dist = 50

ax.view_init(15, 120)
fig
