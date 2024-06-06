import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import scipy.stats as stats  # 用于统计函数
from scipy.stats import scoreatpercentile  # 用于计算分位数

# 生成数据
def generate_data(a, loc, scale, size, lower, upper):
    data = stats.skewnorm.rvs(a=a, loc=loc, scale=scale, size=size)
    # 确保数据在范围内
    data = data[(data >= lower) & (data <= upper)]
    # 如果数据点不足，继续生成直到达到所需数量
    while len(data) < size:
        additional_data = stats.skewnorm.rvs(a=a, loc=loc, scale=scale, size=(size - len(data)))
        additional_data = additional_data[(additional_data >= lower) & (additional_data <= upper)]
        data = np.concatenate((data, additional_data))
    return data[:size]

morning_data = generate_data(a=0.15, loc=42, scale=10, size=9000, lower=0, upper=58)
afternoon_data = generate_data(a=1, loc=28, scale=10, size=700, lower=0, upper=53)

morning_mean = np.mean(morning_data)  # 平均值
afternoon_mean = np.mean(afternoon_data)
mean_diff = morning_mean - afternoon_mean  # 差值

# 匹配中位数和其他分位数
morning_sorted = np.sort(morning_data)  # 排序
afternoon_sorted = np.sort(afternoon_data)

# 遍历0到100的分位数
for percentile in np.linspace(0, 100, 101):
    morning_value = scoreatpercentile(morning_sorted, percentile)
    afternoon_value = scoreatpercentile(afternoon_sorted, percentile)
    adjustment = morning_value - afternoon_value
    afternoon_sorted[np.isclose(afternoon_sorted, afternoon_value, atol=1e-6)] += adjustment

# 调整下午平均值，使其与上午平均值相差2以内
if abs(mean_diff) > 2:
    adjustment = mean_diff - 2 if mean_diff > 0 else mean_diff + 2
    afternoon_data = afternoon_data + adjustment

# 确保调整后的数据在正确范围内
afternoon_sorted = np.clip(afternoon_sorted, 0, 60)

# 合并上午数据和调整后的下午数据
combined_data = np.concatenate((morning_sorted, afternoon_sorted))

# 绘制
plt.hist(morning_sorted, bins=50, density=True, alpha=0.6, color='g', label='Morning')
plt.hist(afternoon_sorted, bins=50, density=True, alpha=0.6, color='b', label='Afternoon (Adjusted)')
plt.hist(combined_data, bins=50, density=True, alpha=0.6, color='r', label='Combined')
plt.title('Combined Exam Score Distribution')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()