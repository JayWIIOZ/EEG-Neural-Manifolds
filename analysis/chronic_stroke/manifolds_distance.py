import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from pyriemann.utils.distance import distance_riemann
from scipy.linalg import orthogonal_procrustes
from dtaidistance import dtw

# 假设:
# pre_manifolds: 前7个流形矩阵的列表 (每个30×18)
# post_manifolds: 后7个流形矩阵的列表 (每个30×18)
# pre_scores: 康复前分数列表 (长度7)
# post_scores: 康复后分数列表 (长度7)

def pca_distance_metric(pre_matrices, post_matrices, n_components=5):
    """计算PCA空间中的流形距离"""
    distances = []
    for pre, post in zip(pre_matrices, post_matrices):
        # 合并前后数据
        combined = np.vstack((pre, post))

        # PCA降维
        pca = PCA(n_components=n_components)
        pca.fit(combined)

        # 转换前后数据
        pre_pca = pca.transform(pre)
        post_pca = pca.transform(post)

        # 计算质心距离
        pre_centroid = np.mean(pre_pca, axis=0)
        post_centroid = np.mean(post_pca, axis=0)

        # 欧氏距离
        dist = np.linalg.norm(pre_centroid - post_centroid)
        distances.append(dist)

    return np.array(distances)

def procrustes_distance(pre_matrices, post_matrices):
    """计算Procrustes距离"""
    distances = []
    for pre, post in zip(pre_matrices, post_matrices):
        # 中心化数据
        pre_centered = pre - np.mean(pre, axis=0)
        post_centered = post - np.mean(post, axis=0)

        # Procrustes分析
        R, scale = orthogonal_procrustes(pre_centered, post_centered)

        # 旋转后的矩阵
        rotated = scale * pre_centered @ R

        # 计算Frobenius范数距离
        dist = np.linalg.norm(rotated - post_centered, 'fro')
        distances.append(dist)

    return np.array(distances)

def grassmann_distance(pre_matrices, post_matrices):
    """计算Grassmann流形上的弦距离"""
    distances = []
    for pre, post in zip(pre_matrices, post_matrices):
        # 对每个矩阵进行SVD分解
        U_pre, _, _ = np.linalg.svd(pre, full_matrices=False)
        U_post, _, _ = np.linalg.svd(post, full_matrices=False)

        # 计算弦距离
        M = U_pre.T @ U_post
        dist = np.sqrt(np.minimum(pre.shape[1], post.shape[1]) - np.linalg.norm(M, 'fro') ** 2)
        distances.append(dist)

    return np.array(distances)

def dtw_distance(pre_matrices, post_matrices):
    """使用DTW计算时间序列距离"""
    distances = []
    for pre, post in zip(pre_matrices, post_matrices):
        # 计算质心轨迹
        pre_centroid = np.mean(pre, axis=1)
        post_centroid = np.mean(post, axis=1)

        # 计算DTW距离
        dist = dtw.distance(pre_centroid, post_centroid)
        distances.append(dist)

    return np.array(distances)

def riemannian_distance(pre_matrices, post_matrices):
    """计算黎曼几何距离 (修正版)"""
    distances = []
    for pre, post in zip(pre_matrices, post_matrices):
        # 计算协方差矩阵 (确保正定)
        cov_pre = np.cov(pre, rowvar=False) + 1e-6 * np.eye(pre.shape[1])
        cov_post = np.cov(post, rowvar=False) + 1e-6 * np.eye(post.shape[1])

        # 计算黎曼距离 (新版调用方式)
        dist = distance_riemann(cov_pre, cov_post)
        distances.append(dist)

    return np.array(distances)
