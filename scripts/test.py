import numpy as np
import matplotlib.pyplot as plt

# 定义参数范围
# t = np.linspace(0, 15, 1000)
# k = 0.42 # 0.42 - 0.5, lower, 1500 steps

# t = np.linspace(0, 5.5, 1000)
# k = 1.2 # 1.2 - 1.4, normal, 550 steps

t = np.linspace(0, 3.5, 1000)
k = 1.8 # 1.8 - 1.9, fast, 350 steps

# 定义参数方程
x = np.sin(k * t)
y = np.sin(k * t) * np.cos(k * t)

# 计算速度
dx_dt = k * np.cos(k * t)
dy_dt = k * np.cos(k * t)**2 - k * np.sin(k * t)**2
speed = np.sqrt(dx_dt**2 + dy_dt**2)

# 创建图形
plt.figure(figsize=(8, 4))

# 绘制速度随时间变化的图形
plt.plot(t, speed, label='Speed')
# plt.plot(x, y)

# 设置图形属性
plt.title('Speed vs Time')
plt.xlabel('Time (t)')
plt.ylabel('Speed')
plt.legend()
plt.grid(True)

# 显示图形
plt.savefig('8')