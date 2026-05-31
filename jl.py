"""
聚类分析实验报告
数据集：Iris数据集
算法：K-means聚类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# 1. 数据来源
print("1. 数据来源")
print("使用sklearn内置的Iris数据集，这是一个经典的多变量数据集，包含150个样本，每个样本有4个特征。")
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# 2. 数据预处理
print("\n2. 数据预处理")
# 2.1 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2.2 数据可视化
plt.figure(figsize=(12, 5))

# 2.2.1 特征分布图
plt.subplot(1, 2, 1)
sns.boxplot(data=pd.DataFrame(X, columns=feature_names))
plt.xticks(rotation=45)
plt.title('特征分布箱线图')

# 2.2.2 特征相关性热力图
plt.subplot(1, 2, 2)
sns.heatmap(pd.DataFrame(X, columns=feature_names).corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.savefig('iris_preprocessing.png')
plt.close()

# 3. 聚类分析
print("\n3. 聚类分析")
# 3.1 确定最佳聚类数
inertias = []
silhouette_scores = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# 3.2 绘制肘部法则图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K, inertias, 'bx-')
plt.xlabel('k值')
plt.ylabel('惯性')
plt.title('肘部法则图')

# 3.3 绘制轮廓系数图
plt.subplot(1, 2, 2)
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k值')
plt.ylabel('轮廓系数')
plt.title('轮廓系数图')
plt.tight_layout()
plt.savefig('k_selection.png')
plt.close()

# 3.4 执行最终聚类
best_k = 3  # 根据肘部法则和轮廓系数确定
kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. 结果分析
print("\n4. 结果分析")
# 4.1 聚类结果可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           marker='x', s=200, linewidths=3, color='r', label='聚类中心')
plt.xlabel('标准化后的特征1')
plt.ylabel('标准化后的特征2')
plt.title('K-means聚类结果')
plt.legend()
plt.savefig('clustering_results.png')
plt.close()

# 4.2 输出聚类评估指标
print(f"\n聚类评估指标：")
print(f"轮廓系数: {silhouette_score(X_scaled, clusters):.3f}")
print(f"惯性: {kmeans.inertia_:.3f}")

# 4.3 分析每个簇的特征
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_),
                             columns=feature_names)
print("\n各簇中心点特征值：")
print(cluster_centers)

# 4.4 计算每个簇的样本数
cluster_counts = pd.Series(clusters).value_counts().sort_index()
print("\n各簇样本数量：")
print(cluster_counts)
