import numpy as np  # 计算
import matplotlib.pyplot as plt  # 绘图
from scipy.stats import skewnorm, scoreatpercentile  # 统计&计算分位数
from sklearn.mixture import GaussianMixture  # 高斯混合模型
import random as ran  # 生成随机数


# 生成数据
def generate_data(a, loc, scale, size, lower, upper):
    data = skewnorm.rvs(a=a, loc=loc, scale=scale, size=size)
    data = data[(data >= lower) & (data <= upper)]  # 确保数据在范围内
    # 如果数据点不足，继续生成直到达到所需数量
    while len(data) < size:
        additional_data = skewnorm.rvs(a=a, loc=loc, scale=scale, size=(size - len(data)))
        additional_data = additional_data[(additional_data >= lower) & (additional_data <= upper)]
        data = np.concatenate((data, additional_data))
    return data[:size]


a_random = round(ran.uniform(0.01, 0.3), 2)  # a值随机
morning_data = generate_data(a=-a_random, loc=42, scale=10, size=9000, lower=0, upper=58)
afternoon_data = generate_data(a=a_random, loc=28, scale=10, size=700, lower=0, upper=53)

# 分位数调整
morning_sorted = np.sort(morning_data)
afternoon_sorted = np.sort(afternoon_data)

for percentile in np.linspace(0, 100, 101):
    morning_value = scoreatpercentile(morning_sorted, percentile)
    afternoon_value = scoreatpercentile(afternoon_sorted, percentile)
    adjustment = morning_value - afternoon_value
    afternoon_sorted[np.isclose(afternoon_sorted, afternoon_value, atol=1e-6)] += adjustment

# 调整下午平均值，使其与上午平均值相差2以内
morning_mean = np.mean(morning_sorted)
afternoon_mean = np.mean(afternoon_sorted)
mean_diff = morning_mean - afternoon_mean

if abs(mean_diff) > 2:
    adjustment = mean_diff - 2 if mean_diff > 0 else mean_diff + 2
    afternoon_sorted += adjustment

# 确保调整后的数据在正确范围内
afternoon_sorted = np.clip(afternoon_sorted, 0, 60)


# 创建高斯混合模型并进行数据拟合
def fit_gmm(data, n_components):
    # n_components表示处理精度，covariance_type='full'表示使用全协方差矩阵，random_state=42设置随机种子（确保结果的可重复性）
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(data.reshape(-1, 1))  # 高斯混合模型需要二维输入，将数据调整为二维数组
    return gmm


# 调整下午数据的分布以匹配上午数据
def adjust_data_distribution(morning_data, afternoon_data, n_components):
    morning_gmm = fit_gmm(morning_data, n_components)
    afternoon_gmm = fit_gmm(afternoon_data, n_components)

    morning_sampled, _ = morning_gmm.sample(len(afternoon_data))  # 从早上模型中采样，生成与下午数据长度相同的相似分布
    afternoon_adjusted = morning_sampled.flatten()  # 将采样的数据调整为一维数组

    afternoon_adjusted = np.clip(afternoon_adjusted, np.min(afternoon_data),
                                 np.max(afternoon_data))  # 将调整后的数据限制在下午数据的最小值和最大值之间

    return morning_gmm, afternoon_gmm, afternoon_adjusted


morning_gmm, afternoon_gmm, afternoon_adjusted = adjust_data_distribution(morning_sorted, afternoon_sorted,
                                                                          n_components=5)

# 最低分为0
morning_sorted[0] = 0
afternoon_adjusted[0] = 0

# 合并上午数据和调整后的下午数据
combined_data = np.concatenate((morning_sorted, afternoon_adjusted))

# 绘制
# plt.hist(morning_sorted, bins=100, density=True, alpha=0.6, color='g', label='Morning')
# plt.hist(afternoon_adjusted, bins=100, density=True, alpha=0.6, color='b', label='Afternoon (Adjusted)')
plt.hist(combined_data, bins=100, density=True, alpha=0.6, color='r', label='Combined')
plt.title('Combined Exam Score Distribution a=' + str(a_random))
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
