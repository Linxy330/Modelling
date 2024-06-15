import numpy as np  # 计算
import matplotlib.pyplot as plt  # 绘图
from scipy.stats import skewnorm, scoreatpercentile  # 统计&计算分位数
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
    data = np.round(data).astype(int)
    return data[:size]


# 生成数据
a_random_1 = round(ran.uniform(-0.1, 0.1), 3)  # a值随机
a_random_2 = round(ran.uniform(-0.1, 0.1), 3)
morning_data = generate_data(a=a_random_1, loc=42, scale=10, size=9000, lower=0, upper=58)
afternoon_data = generate_data(a=a_random_2, loc=28, scale=10, size=700, lower=0, upper=53)

morning_sorted = np.sort(morning_data)
afternoon_sorted = np.sort(afternoon_data)

# 调整下午平均值，使其与上午平均值相差2以内
morning_mean = np.mean(morning_sorted)
afternoon_mean = np.mean(afternoon_sorted)
mean_diff = morning_mean - afternoon_mean

if abs(mean_diff) > 2:
    adjustment = mean_diff - 2 if mean_diff > 0 else mean_diff + 2
    addition_random = round(ran.uniform(0, 4))
    afternoon_sorted += int(adjustment) + addition_random

# 分位数调整
for percentile in np.linspace(0, 100, 7):
    morning_value = int(scoreatpercentile(morning_sorted, percentile))
    afternoon_value = int(scoreatpercentile(afternoon_sorted, percentile))
    adjustment = morning_value - afternoon_value
    print(str(morning_value) + ' ' + str(afternoon_value))
    percentiles = np.isclose(afternoon_sorted, afternoon_value, atol=0.01)
    afternoon_sorted[percentiles] += adjustment

afternoon_sorted[afternoon_sorted > 60] = 60

afternoon_adjusted = afternoon_sorted

# 合并上午数据和调整后的下午数据
combined_data = np.concatenate((morning_sorted, afternoon_adjusted))

# 统计平均值
morning_mean = round(np.mean(morning_sorted), 2)
afternoon_origin_mean = round(np.mean(afternoon_data), 2)
afternoon_mean = round(np.mean(afternoon_sorted), 2)
combined_mean = round(np.mean(combined_data), 2)
# 统计人数
morning_len = len(morning_sorted)
afternoon_origin_len = len(afternoon_data)
afternoon_len = len(afternoon_sorted)
combined_len = len(combined_data)

# 绘制
plt.hist(morning_sorted, bins=50, density=True, alpha=0.6, color='g',
         label='Morning mean=' + str(morning_mean) + ' len=' + str(morning_len))
plt.hist(afternoon_data, bins=50, density=True, alpha=0.6, color='y',
         label='Afternoon mean=' + str(afternoon_origin_mean) + ' len=' + str(afternoon_origin_len))
plt.hist(afternoon_adjusted, bins=50, density=True, alpha=0.6, color='b',
         label='Afternoon (Adjusted) mean=' + str(afternoon_mean) + ' len=' + str(afternoon_len))
plt.hist(combined_data, bins=50, density=True, alpha=0.6, color='r',
         label='Combined mean=' + str(combined_mean) + ' len=' + str(combined_len))
plt.title('Combined Exam Score Distribution a1=' + str(a_random_1) + ' a2=' + str(a_random_2))
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()
