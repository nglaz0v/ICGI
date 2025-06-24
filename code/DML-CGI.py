from sklearnex import patch_sklearn  # 加速 scikit-learn
patch_sklearn()  # 应用性能优化

import os
import math
import json
import random
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from pyHSICLasso import HSICLasso

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

# warnings.filterwarnings("ignore")


# 固定随机种子。确保涉及到随机操作部分代码执行结果的可重复性和可预测性。
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def pre_processing(X_df, choice="z-score"):
    # 提供统一的数据规范化接口, 实现常用的数据规范化方法, 封装为一个可扩展的函数。
    # 方便后续调用和加入新的方法 (例如分位数标准化、L2 归一化、对数处理、高斯秩变换等)
    if choice == "z-score":
        scaler = StandardScaler()
    elif choice == "min-max":
        scaler = MinMaxScaler()

    return scaler.fit_transform(X_df)


def hsic_lasso(X_df, y):
    """解决生物信息学中高维基因数据的特征选择问题
    Hilbert Schmidt Independence Criterion Lasso (HSIC Lasso)
    1. Can efficiently find nonlinearly related features.
    2. Can find non-redundant features.
    3. Can obtain a globally optimal solution.
    """
    hsic_lasso = HSICLasso()

    genes_name = X_df.columns.tolist()
    # 输入：特征矩阵数据、标签向量以及特征名称
    hsic_lasso.input(pre_processing(X_df, "z-score"), y.values, featname=genes_name)

    # 参数 M: M 是随机排列参数   使用 bagging 提高选择稳定性
    # 参数 B: B << n 为块参数   将训练数据集划分为 n/B 个分区   B 就是每个块中的样本数
    hsic_lasso.classification(
        num_feat=50, B=30, M=3, discrete_x=False, max_neighbors=10, n_jobs=-1
    )
    # 输出：特征选择结果
    index_score = hsic_lasso.get_index_score()
    print(index_score, len(index_score))

    top_features = [g for g in hsic_lasso.get_features()]

    return top_features


# 创建了一个 ArgumentParser 对象, 然后, 我们调用 add_argument() 方法添加一个命令行参数。
args = argparse.ArgumentParser()
args.add_argument(
    "--dataset", type=str, choices=["LUAD", "LUSC", "BLCA", "BRCA", "KIRC", "LIHC"]
)
# 我们调用 parse_args() 方法来解析命令行参数。这个方法会返回一个命名空间, 我们可以使用 args.attr 的方式来获取指定的变量。
args = args.parse_args()

# 通过命令行参数灵活切换癌症数据集，自动创建日志目录并追加记录。
log_file_path = "./logs/log_20240227.txt"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
log_file = open(log_file_path, "a+")
dataset, gene_expression = args.dataset, "TPM"
print("Dataset name:", dataset, file=log_file)
print("Dataset name:", dataset)

# 数据加载和特征选择。当设置 index_col=0 时, 它指定 DataFrame 中的第一列作为索引列
df = pd.read_csv(f"./dataset/{dataset}_{gene_expression}.csv", index_col=0)
# 基因表达数据矩阵和标签
X, y = df.iloc[:, :-1], df["label"]
print(X, y)

start = datetime.now()

# 使用 HSIC Lasso 和 CFS-MI 两种方法初步筛选候选基因
# HSIC Lasso
hsic_genes = hsic_lasso(X.copy(deep=True), y)  # 显式拷贝避免原始数据污染
print("Block HISC Lasso:", hsic_genes, len(hsic_genes), file=log_file)
print("Block HISC Lasso:", hsic_genes, len(hsic_genes))

# CFS-MI
with open(f"./Exp/CFS/CFS_{dataset}.txt", "r") as f:
    # Matlab 中的索引从 1 开始, 而 Python 中的索引从 0 开始, 因此需要减去 1。
    cfs_idx = [int(idx) - 1 for idx in f.read().split("\n")]
cfs_genes = [gene for gene in X.iloc[:, cfs_idx].columns]
print("CFS-MI:", cfs_genes, len(cfs_genes), file=log_file)
print("CFS-MI:", cfs_genes, len(cfs_genes))

print(
    "Block HISC Lasso:",
    len({gene for gene in hsic_genes} - {gene for gene in cfs_genes}),
    file=log_file,
)
print(
    "CFS-MI:",
    len({gene for gene in cfs_genes} - {gene for gene in hsic_genes}),
    file=log_file,
)
print(
    "Intersection of genes:",
    len({gene for gene in cfs_genes} & {gene for gene in hsic_genes}),
    file=log_file,
)

# 因果特征选择, 取并集, 在大域范围保留更多真阳性。
union_genes = {gene for gene in cfs_genes} | {gene for gene in hsic_genes}
candidate_genes = [gene for gene in union_genes]
candidate_genes.sort()
print("Candidate genes:", candidate_genes, len(candidate_genes), file=log_file)
print("Candidate genes:", candidate_genes, len(candidate_genes))

X1, y = df[candidate_genes].copy(deep=True), df["label"].values
X1 = pre_processing(X1, "z-score")  # Z-Score 标准化  均值为 0, 方差为 1
n = X1.shape[0]  # 样本数
# print(X1, "\n", y)

# 标签类别不平衡处理
y_0 = np.where(y == 0)[0]  # 标签为 0 的索引数组
y_1 = np.where(y == 1)[0]  # 标签为 1 的索引数组
# 需要随机打乱运行多次实验?
# np.random.shuffle(y_0)
# np.random.shuffle(y_1)
split_0 = len(y_0) // 2
split_1 = len(y_1) // 2
# 拼接一下两个一维索引数组
index1 = np.concatenate((y_0[:split_0], y_1[:split_1]))
index2 = np.concatenate((y_0[split_0:], y_1[split_1:]))

causal_genes = []
# IC: index1   I: index2
print("-" * 100)
# 基于去偏机器学习（Debiased Machine Learning, DML）和交叉拟合策略进行假设的因果效应估计
for col_index in range(0, X1.shape[1]):  # one-vs.-the-rest search
    result_dict = {"Gene": candidate_genes[col_index]}
    res_cols = [i for i in range(0, X1.shape[1]) if i != col_index]
    z = X1[:, res_cols]  # 控制变量
    d = X1[:, col_index]  # 待选择变量

    # 随机森林估计器  隐式正则化  注意防止过拟合
    model1 = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        ccp_alpha=0.01,
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model2 = RandomForestClassifier(
        n_estimators=200,
        criterion="gini",
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        ccp_alpha=0.01,
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model1.fit(z[index1, :], y[index1])
    model2.fit(z[index2, :], y[index2])
    G1 = model1.predict_proba(z[index2, :])[:, 1]
    G2 = model2.predict_proba(z[index1, :])[:, 1]

    model1 = RandomForestRegressor(
        n_estimators=200,
        criterion="squared_error",
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="log2",
        ccp_alpha=0.01,
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model2 = RandomForestRegressor(
        n_estimators=200,
        criterion="squared_error",
        max_depth=6,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features="log2",
        ccp_alpha=0.01,
        bootstrap=True,
        max_samples=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model1.fit(z[index1, :], d[index1])
    model2.fit(z[index2, :], d[index2])
    M1 = model1.predict(z[index2, :])
    M2 = model2.predict(z[index1, :])
    V1 = d[index2] - M1
    V2 = d[index1] - M2

    # 计算 Cross-Fitting Debiased ML 的 theta
    theta1 = np.mean(V1 * (y[index2] - G1)) / np.mean(V1 * d[index2])
    theta2 = np.mean(V2 * (y[index1] - G2)) / np.mean(V2 * d[index1])
    theta_cf = (theta1 + theta2) / 2

    # Direct way 计算交叉拟合得到的 Khi 和 Sigma
    khi1 = np.mean((G1 - y[index2]) * (d[index2] - M1))
    khi2 = np.mean((G2 - y[index1]) * (d[index1] - M2))
    sigmatwo1 = np.mean(((G1 - y[index2]) * (d[index2] - M1) - khi1) ** 2)
    sigmatwo2 = np.mean(((G2 - y[index1]) * (d[index1] - M2) - khi2) ** 2)
    khihat = (khi1 + khi2) / 2
    sigmatwohat = (sigmatwo1 + sigmatwo2) / 2

    # 通过正态性检验和 p 值计算确定因果基因
    p_value = 2 * norm.sf(abs(khihat), 0, math.sqrt(sigmatwohat / n))
    # p_value = 2 * norm.sf(abs(khihat), 0, math.sqrt(sigmatwohat) / math.sqrt(n))

    is_parent_flag = 0 if p_value > 0.05 else 1
    if is_parent_flag == 1:  # 原假设: 不是因果特征, 拒绝原假设, 则是因果特征。
        causal_genes.append(candidate_genes[col_index])

    result_dict["Thetahat"] = theta_cf
    result_dict["pValue"] = p_value
    result_dict["is_parent"] = is_parent_flag

    print(result_dict)

print("Dataset name:", dataset)
print(causal_genes, len(causal_genes))
print(causal_genes, len(causal_genes), file=log_file)
print("\n")
print("\n", file=log_file)
delta = (datetime.now() - start).total_seconds()
print("用时：{:.6f}s".format(delta))
print("-" * 100)

# 将识别结果保存为结构化 JSON 文件
with open("./results/causal_genes_ci.json", "r") as f:
    genes_json = json.load(f)
genes_json["Ours"][dataset] = causal_genes
json_str = json.dumps(genes_json, ensure_ascii=False, indent=2)
with open("./results/causal_genes_ci.json", "w") as f:
    f.write(json_str)
