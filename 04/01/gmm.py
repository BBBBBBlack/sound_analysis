import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mul_nor_1
from numpy.random import multivariate_normal as mul_nor_2
from sklearn.cluster import KMeans


def init_params(shape, K):
    N, D = shape
    mu = np.random.rand(K, D)
    cov = np.array([np.eye(D)] * K)
    alpha = np.array([1.0 / K] * K)
    print("Parameters initialized.")
    print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


def phi(Y, mu_k, cov_k):
    norm = mul_nor_1(mean=mu_k, cov=cov_k)
    return norm.pdf(Y)


def getExpectation(Y, mu, cov, alpha):
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = alpha.shape[0]
    assert N > 1, "There must be more than one sample!"
    assert K > 1, "There must be more than one gaussian model!"
    # 响应度矩阵
    gamma = np.mat(np.zeros((N, K)))
    # 计算各模型中所有样本出现的概率
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], cov[k])
    prob = np.mat(prob)
    # 计算每个模型对每个样本的响应度
    for k in range(K):
        gamma[:, k] = alpha[k] * prob[:, k]
    for i in range(N):
        gamma[i, :] /= np.sum(gamma[i, :])
    return gamma


def maximize(Y, gamma):
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    # 初始化参数值
    mu = np.zeros((K, D))
    cov = []
    alpha = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        mu[k, :] = np.sum(np.multiply(Y, gamma[:, k]), axis=0) / Nk
        # 更新 cov
        cov_k = (Y - mu[k]).T * np.multiply((Y - mu[k]), gamma[:, k]) / Nk
        cov.append(cov_k)
        # 更新 alpha
        alpha[k] = Nk / N
    cov = np.array(cov)
    return mu, cov, alpha


def GMM_EM(Y, K, times):
    mu, cov, alpha = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, cov, alpha)
        mu, cov, alpha = maximize(Y, gamma)
    print("{sep} Result {sep}".format(sep="-" * 20))
    print("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
    return mu, cov, alpha


def gen_data():
    cov1 = np.mat("0.3 0;0 0.1")
    cov2 = np.mat("0.2 0;0 0.3")

    mu1 = np.array([0, 1])
    mu2 = np.array([2, 1])

    sample = np.zeros((100, 2))
    sample[:30, :] = mul_nor_2(mean=mu1, cov=cov1, size=30)
    sample[30:, :] = mul_nor_2(mean=mu2, cov=cov2, size=70)
    np.savetxt("sample.data", sample)

    plt.plot(sample[:30, 0], sample[:30, 1], "bo")
    plt.plot(sample[30:, 0], sample[30:, 1], "rs")
    plt.show()


def gen_datas2():
    cov1 = np.mat("0.3 0;0 0.1")
    cov2 = np.mat("0.2 0;0 0.3")
    cov3 = np.mat("0.1 0;0 0.2")
    cov4 = np.mat("0.3 0;0 0.3")

    mu1 = np.array([0, 1])
    mu2 = np.array([2, 1])
    mu3 = np.array([-1, -2])
    mu4 = np.array([1, -1])

    sample = np.zeros((200, 2))
    sample[:50, :] = mul_nor_2(mean=mu1, cov=cov1, size=50)
    sample[50:100, :] = mul_nor_2(mean=mu2, cov=cov2, size=50)
    sample[100:150, :] = mul_nor_2(mean=mu3, cov=cov3, size=50)
    sample[150:, :] = mul_nor_2(mean=mu4, cov=cov4, size=50)
    np.savetxt("sample.data", sample)
    plt.plot(sample[:50, 0], sample[:50, 1], "bs")
    plt.plot(sample[50:100, 0], sample[50:100, 1], "rs")
    plt.plot(sample[100:150, 0], sample[100:150, 1], "gs")
    plt.plot(sample[150:, 0], sample[150:, 1], "ys")
    plt.show()


gen_datas2()


def scale_data(Y):
    # 对每一维特征分别进行缩放
    for i in range(Y.shape[1]):
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y


def test_gauss():
    # 载入数据
    Y = np.loadtxt("sample.data")
    matY = np.matrix(Y, copy=True)
    # 模型个数，即聚类的类别个数
    K = 4
    # 数据预处理
    Y = scale_data(Y)
    # 计算 GMM 模型参数
    mu, cov, alpha = GMM_EM(matY, K, 100)
    # 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
    N = Y.shape[0]
    # 求当前模型参数下，各模型对样本的响应度矩阵
    gamma = getExpectation(matY, mu, cov, alpha)
    # 对每个样本，求响应度最大的模型下标，作为其类别标识
    category = gamma.argmax(axis=1).flatten().tolist()[0]
    # 将每个样本放入对应类别的列表中
    class1 = np.array([Y[i] for i in range(N) if category[i] == 0])
    class2 = np.array([Y[i] for i in range(N) if category[i] == 1])
    class3 = np.array([Y[i] for i in range(N) if category[i] == 2])
    class4 = np.array([Y[i] for i in range(N) if category[i] == 3])

    # 绘制聚类结果
    plt.figure()
    plt.plot(class1[:, 0], class1[:, 1], 'bo', label="class1")
    plt.plot(class2[:, 0], class2[:, 1], 'ro', label="class2")
    plt.plot(class3[:, 0], class3[:, 1], 'go', label="class3")
    plt.plot(class4[:, 0], class4[:, 1], 'yo', label="class4")
    plt.legend(loc="best")
    plt.title("GMM Clustering By EM Algorithm")

    # K-means 算法
    # 聚类为 4 类
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(Y)
    # 获取聚类标签
    pred_label = kmeans.labels_
    # 获取聚类中心
    pred_center = kmeans.cluster_centers_
    # 绘制聚类结果
    plt.figure()
    plt.plot(Y[pred_label == 0][:, 0], Y[pred_label == 0][:, 1],
             'bo', label="class1")
    plt.plot(Y[pred_label == 1][:, 0], Y[pred_label == 1][:, 1],
             'ro', label="class2")
    plt.plot(Y[pred_label == 2][:, 0], Y[pred_label == 2][:, 1],
             'go', label="class3")
    plt.plot(Y[pred_label == 3][:, 0], Y[pred_label == 3][:, 1],
             'yo', label="class4")
    plt.legend(loc="best")
    plt.title("K-means Clustering")
    plt.show()


test_gauss()
