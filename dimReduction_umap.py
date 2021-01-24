#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 12:03:21 2021

@author: aikenchung
For running 3D UMAP analysis on MN, FSGS, and IgAN datasets.
"""

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import umap as umap2

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

diseaseType = 'GN3D' # GN3D-73, IgAN-27, FSGS-25, MN-21
inputGeneList = 'ER271'
filePath = 'umap_'+diseaseType+'_'+inputGeneList+'.csv'
# load dataset into Pandas DataFrame
geneExprData = pd.read_csv(filePath)
geneExprData.head()
nrow, ncol = geneExprData.shape
healthyCtrl = 21

reducer = umap.UMAP()

geneExpr_Data = geneExprData.iloc[:,1:geneExprData.shape[1]].values

scaled_geneExpr_Data = StandardScaler().fit_transform(geneExpr_Data)

embedding = reducer.fit_transform(scaled_geneExpr_Data)
embedding.shape

targets = ['Healthy', 'IgAN','FSGS','MN']
colors = ['b', '','','orange']

fig0 = plt.figure(figsize = (8,8))
plot0 = fig0.add_subplot(1,1,1)
for target, color in zip(targets,colors):
    indicesToKeep = geneExprData['Group'] == target
    plot0.scatter(
        embedding[indicesToKeep, 0],
        embedding[indicesToKeep, 1],
        c=color)
plot0.set_title('2D UMAP projection of JuCKD ' +diseaseType+' dataset', fontsize=24)
plot0.legend(targets)
########################################################

reducer2 = umap2.UMAP(n_components=3)

embedding2 = reducer2.fit_transform(scaled_geneExpr_Data)
embedding2.shape

fig1 = plt.figure(figsize = (10,8))
plot1 = fig1.add_subplot(111, projection='3d')
plot1.set_xlabel('\nComponent-1', fontsize = 15)
plot1.set_ylabel('\nComponent-2', fontsize = 15)
plot1.set_zlabel('\nComponent-3', fontsize = 15)
plot1.tick_params(axis='x', which='major', pad=0.01)
plt.rc('xtick',labelsize=14)

for target, color in zip(targets,colors):
    indicesToKeep = geneExprData['Group'] == target
    plot1.scatter(
        embedding2[indicesToKeep, 0],
        embedding2[indicesToKeep, 1],
        embedding2[indicesToKeep, 2],
        c=color,
        depthshade = False)
plot1.set_title('3D UMAP projection of JuCKD ' +diseaseType+' dataset', fontsize=24)
plot1.legend(targets)
plot1.view_init(10, 60)
fig1
