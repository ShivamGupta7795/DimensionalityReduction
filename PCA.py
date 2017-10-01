"""
course: 		CSE 601 - Data Mining
date: 			09/28/09
developed by:	Jay Bakshi
				Shivam Gupta
				Debanjan Paul
filename:		PCA.py
version:		1.0
description: 	This program performs dimensionality reduction using PCA, SVD and t-SNE algorithms.
libraries:		numpy	
				matplotlib
				scikit-klearn
				pandas				
"""

################################################################################
# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from random import randint
import sys

################################################################################
# File name taken as a command line parameter.
file = sys.argv[1]
################################################################################
#  Imports the tab delimited file excluding the last column into a numpy matrix
data=np.genfromtxt(file, delimiter="\t")[:,:-1]

################################################################################
#  Import the labels into a list, extract using pandas only the last column
labels=list(pd.read_csv(file, sep='\t', lineterminator='\n', header=None).iloc[:,-1])

################################################################################
#  Normalizing the data
X=(data - data.mean(0))

################################################################################
#  Compute the covariance matrix
S=(1/(X.shape[0]))*X.T.dot(X)

################################################################################
#  Compute and extract the eigen vectors from the covariance matrix
eigen_vectors=np.linalg.eig(S)[1]

################################################################################
#  Select the first two columns from the eigen vector table as the principal components
#  recompute samples based on principal components
pca_plotData=data.dot(eigen_vectors[:,0:2])

################################################################################
#  Map labels from strings into integers
lb=list(set(labels))
label={}
for i in range(len(lb)):
    label[lb[i]]=i

################################################################################
#  Transform resulting <i>plotData</i> and <i>label</i> into one single dataframe
df_pca = pd.DataFrame(dict(x=list(pca_plotData[:,0]),y=list(pca_plotData[:,1]), labels=labels))

################################################################################
#  Plotting the dataframe with labels
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i in range(len(lb)):
    ax1.scatter(df_pca[df_pca['labels']==lb[i]]['x'], df_pca[df_pca['labels']==lb[i]]['y'], color=("#"+str(randint(1000000, 9999999))[:-1]), label=lb[i])
plt.legend(loc='upper left')
plt.title("PCA plot on "+file)
plt.savefig('PCA_plots_'+file[:-4]+'.pdf')

################################################################################
# SVD
U = np.linalg.svd(X.T)[0][:,:2]
svd_plotData = data.dot(U)
df_svd = pd.DataFrame(dict(x=list(svd_plotData[:,0]),y=list(svd_plotData[:,1]), labels=labels))
fig = plt.figure()
ax2 = fig.add_subplot(111)
for i in range(len(lb)):
    ax2.scatter(df_svd[df_svd['labels']==lb[i]]['x'], df_svd[df_svd['labels']==lb[i]]['y'], color=("#"+str(randint(1000000, 9999999))[:-1]), label=lb[i])
plt.legend(loc='upper left')
plt.title("SVD plot on "+file)
plt.savefig('SVD_plots_'+file[:-4]+'.pdf')
################################################################################
# t-SNE
X_embedded = TSNE(n_components=2).fit_transform(X.T)
tsne_plotData = data.dot(X_embedded)
df_tsne = pd.DataFrame(dict(x=list(tsne_plotData[:,0]),y=list(tsne_plotData[:,1]), labels=labels))
fig = plt.figure()
ax3 = fig.add_subplot(111)
for i in range(len(lb)):
    ax3.scatter(df_tsne[df_tsne['labels']==lb[i]]['x'], df_tsne[df_svd['labels']==lb[i]]['y'], color=("#"+str(randint(1000000, 9999999))[:-1]), label=lb[i])
plt.legend(loc='upper left')
plt.title("t-SNE plot on "+file)
plt.savefig('t-SNE_plots_'+file[:-4]+'.pdf')
plt.show()