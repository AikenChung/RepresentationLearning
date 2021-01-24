# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 16:37:23 2020

@author: Aiken

For running 3D PCA analysis on MN, FSGS, and IgAN datasets.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

filePath = '/filePathTo/file.csv'
# load dataset into Pandas DataFrame
df = pd.read_csv(filePath)
nrow, ncol = df.shape
healthyCtrl = 21 # put the number of sample in the control group

# Standardizing the data
norm_data = StandardScaler().fit_transform(df)

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(norm_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC-1', 'PC-2', 'PC-3'])

pca2 = PCA(n_components=2)
principalComponents2 = pca2.fit_transform(norm_data)
principalDf2 = pd.DataFrame(data = principalComponents2
             , columns = ['PC-1', 'PC-2'])

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

principalDf['Groups'] = group
principalDf2['Groups'] = group

targets = ['Healthy', 'IgAN','FSGS','MN']
colors = ['b', 'r','g','orange']

# for 2D ploting
fig0 = plt.figure(figsize = (8,8))
ax0 = fig0.add_subplot(1,1,1) 
ax0.set_xlabel('PC-1', fontsize = 15)
ax0.set_ylabel('PC-2', fontsize = 15)
ax0.set_title(('PCA Plot'), fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = principalDf2['Groups'] == target
    ax0.scatter(principalDf2.loc[indicesToKeep, 'PC-1']
               , principalDf2.loc[indicesToKeep, 'PC-2']
               , c = color
               , s = 50)
ax0.legend(targets)

# for 3D ploting
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('\nPC-1', fontsize = 15)
ax.set_ylabel('\nPC-2', fontsize = 15)
ax.set_zlabel('\nPC-3', fontsize = 15)
ax.tick_params(axis='x', which='major', pad=0.01)
plt.rc('xtick',labelsize=14)

ax.tick_params(axis='y', which='major', pad=0.01)
ax.tick_params(axis='z', which='major', pad=0.01)
ax.zaxis.labelpad = -5
ax.set_title(('PCA Plot'), fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = principalDf['Groups'] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC-1']
               , principalDf.loc[indicesToKeep, 'PC-2']
               , principalDf.loc[indicesToKeep, 'PC-3']
               , c = color
               , s = 50
               , depthshade = False)
ax.legend(targets)
ax.grid()
ax.dist = 50
