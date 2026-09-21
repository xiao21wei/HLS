import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


import matplotlib.font_manager as fm
font_path = '/data/luopw/production_scheduling/times-new-roman.ttf'
prop = fm.FontProperties(fname=font_path)
plt.ylabel('Makespan', fontproperties=prop)
# 其它文本同理

# 读取数据
df = pd.read_csv('/data/luopw/production_scheduling/results3.csv', header=None)

# 名称
names = df[0]

# makespan 列（第3,5,7,9,11,13,15列，索引为2,4,6,8,10,12,14）
makespans = df[[1,2,4,6,8,10,12,14]]
# PD 列（第4,6,8,10,12,14,16列，索引为3,5,7,9,11,13,15）
pds = df[[3,5,7,9,11,13,15]]

# 绘制names与makespan的散点图 不同的颜色和形状表示不同的算法
plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
plt.scatter(names, makespans.iloc[:, 0], label='UB', color='blue', marker='o')
plt.scatter(names, makespans.iloc[:, 1], label='FIFO+EET', color='orange', marker='s')
plt.scatter(names, makespans.iloc[:, 2], label='FIFO+SPT', color='green', marker='^')
plt.scatter(names, makespans.iloc[:, 3], label='MOPNR+EET', color='red', marker='x')
plt.scatter(names, makespans.iloc[:, 4], label='MOPNR+SPT', color='purple', marker='D')
plt.scatter(names, makespans.iloc[:, 5], label='MWKR+EET', color='brown', marker='v')
plt.scatter(names, makespans.iloc[:, 6], label='MWKR+SPT', color='pink', marker='*')
plt.scatter(names, makespans.iloc[:, 7], label='OURS', color='gray', marker='P')

# plt.xlabel('Names')
plt.ylabel('Makespan', fontproperties=prop)
# plt.title('Makespan vs Names') 
plt.legend(prop=prop)
plt.xticks(rotation=45, ha='right',fontproperties=prop)
plt.yticks(fontproperties=prop)
# plt.grid()
# plt.savefig('makespan_vs_names1.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制names与PD的散点图 不同的颜色和形状表示不同的算法
plt.subplot(1, 2, 2)
plt.scatter(names, pds.iloc[:, 0], label='FIFO+EET', color='orange', marker='s')
plt.scatter(names, pds.iloc[:, 1], label='FIFO+SPT', color='green', marker='^')
plt.scatter(names, pds.iloc[:, 2], label='MOPNR+EET', color='red', marker='x')
plt.scatter(names, pds.iloc[:, 3], label='MOPNR+SPT', color='purple', marker='D')
plt.scatter(names, pds.iloc[:, 4], label='MWKR+EET', color='brown', marker='v')
plt.scatter(names, pds.iloc[:, 5], label='MWKR+SPT', color='pink', marker='*')
plt.scatter(names, pds.iloc[:, 6], label='OURS', color='gray', marker='P')

# plt.xlabel('Names')
plt.ylabel('PD(%)', fontproperties=prop)
# plt.title('PD vs Names')
# plt.legend(prop=prop)
plt.xticks(rotation=45, ha='right',fontproperties=prop)
plt.yticks(fontproperties=prop)
# plt.grid()
plt.savefig('pd_vs_names3.png', dpi=300, bbox_inches='tight')
plt.show()
