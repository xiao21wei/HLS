# 从OUTPUT_DIR路径下读取文件，并从文件最后的两行非空行中提取total time和time cost，保存到excel文件中
# Total time: 42
# Time cost: 41s
# 需要提取出42和41，需要从文件的最后两行非空行中提取，42和41为示例数据，需要匹配字符串中的数字

import pandas as pd
import os

OUTPUT_DIR = '../test_save/drl1'

# 动态生成列名
columns = ['filename', 'total_time1', 'time_cost1', 'total_time2', 'time_cost2', 'total_time3', 'time_cost3', 'total_time4', 'time_cost4' , 'total_time5', 'time_cost5', 'total_time6', 'time_cost6', 'total_time7', 'time_cost7', 'total_time8', 'time_cost8', 'total_time9', 'time_cost9', 'total_time10', 'time_cost10','total_time11', 'time_cost11']
df = pd.DataFrame(columns=columns)
for i in range(1, 12):
    dir = os.path.join(OUTPUT_DIR, str(i))
    for filename in os.listdir(dir):
        if filename.endswith('.txt'):   
            with open(os.path.join(dir, filename), 'r') as f:
                lines = f.readlines()
                # 去除空行
                lines = [line for line in lines if line.strip() != '']
                # 提取最后两行的字符串
                last_two_lines = lines[-2:]
                # 提取total time和time cost
                total_time = last_two_lines[0].split(' ')[-1]
                time_cost = last_two_lines[1].split(' ')[-1]
                if 's' in time_cost:
                    time_cost = time_cost.replace('s', '')
                if total_time and time_cost:
                    # 如果filename不存在于DataFrame中，添加新行
                    if filename not in df['filename'].values:
                        df = pd.concat([df, pd.DataFrame([{'filename': filename}])], ignore_index=True)
                    # 转换为数值类型
                    df.loc[df['filename'] == filename, f'total_time{i}'] = pd.to_numeric(total_time)
                    df.loc[df['filename'] == filename, f'time_cost{i}'] = pd.to_numeric(time_cost)

# df按照filename排序
df = df.sort_values(by='filename')
# 将df保存到excel文件中
df.to_excel('xx.xlsx', index=False)





            
