#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.manifold import TSNE 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tsne = TSNE(n_components=2) 
X_tsne = tsne.fit_transform(train_feature) 
X_tsne_data = np.vstack((X_tsne.T, train_y)).T 
df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1','Dim2','label']) 
df_tsne.head()
plt.figure(figsize=(6, 6)) 
sns.scatterplot(data=df_tsne,hue='label',x='Dim1',y='Dim2')
plt.title('T-SNE visualization of semantic features')
#plt.savefig('T-SNE visualization of semantic features.pdf')
plt.show()


# In[ ]:


import seaborn as sns
from sklearn.decomposition import PCA 
X_pca = PCA(n_components=2).fit_transform(train_X_fea) 
X_pca = np.vstack((X_pca.T, train_y)).T 
df_pca = pd.DataFrame(X_pca, columns=['1st_Component','2n_Component','label']) 
df_pca.head()
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_pca, hue='label',x='1st_Component',y='2n_Component') 
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.title('PCA visualization of semantic features')
plt.savefig('PCA visualization of semantic features.pdf')
plt.show()


# In[ ]:


import numpy as np
import umap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#ss=StandardScaler()
#train_x=ss.fit_transform(train_NAC)          #数据标准化处理

train_y = pd.DataFrame(hs_train_label)
dic = {0:'Negative',1:'Positive'}
ls = []
for index,value in train_y.iterrows():
    arr = np.array(value)[0]
    ls.append(dic[arr])

embedding = umap.UMAP().fit_transform(train_NAC)


plt.figure(figsize=(6, 6))
sns.scatterplot(embedding[:,0],embedding[:,1],hue=ls,palette='Set1',sizes=10)
plt.gca().set_aspect('equal', 'datalim')
plt.title('Umap Visualization of NAC')
plt.xlabel('umap1')
plt.ylabel('umap2')
plt.savefig(r'D:\科研\要做的数据\2021_9_1腺肝数据集\机器学习迭代法\特征编码方案\HS\NAC.png',dpi=300)
plt.show()


# model = torch.load('E:\\myarticle\\best_model\\370.857421875.pth')
#
# # conv1_out = model.conv1(x_train)
# fc_out = fc_out.detach().cpu().numpy().reshape(fc_out.shape[0],-1)
# umap = umap.UMAP(n_components=2)
# fc_out_umap = umap.fit_transform(fc_out)
# fc_out_umap_data = np.vstack((fc_out_umap.T,y_train)).T
# df_umap = pd.DataFrame(fc_out_umap_data, columns=['Dim1', 'Dim2', 'label'])
# df_umap.head()
# plt.figure(figsize=(6, 6))
# sns.scatterplot(data=df_umap, hue='label', x='Dim1', y='Dim2')
# plt.title('Umap Visualization of  fc features')
# plt.legend(loc='best')
# plt.savefig('Umap visualization of fc features.jpg')
# plt.show()


import umap
import  seaborn as sns
umap = umap.UMAP(n_components=2)
X_umap = umap.fit_transform(train_x_fea)
X_umap_data = np.vstack((X_umap.T, train_y)).T
df_umap = pd.DataFrame(X_umap_data, columns=['Dim1','Dim2','label'])
df_umap.head()
plt.figure(figsize=(6, 6))
sns.scatterplot(data=df_umap,hue='label',x='Dim1',y='Dim2')
plt.title('Umap Visualization of features')
plt.legend(loc='best')
plt.savefig('Umap visualization of features.jpg')
plt.show()

