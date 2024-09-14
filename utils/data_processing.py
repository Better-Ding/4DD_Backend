import joblib
import numpy as np
import pandas as pd
import chardet

from utils.response import fail


# 检查文件的编译格式
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


# 将数据处理绘制甘特图需要的数据
def data_to_gantt(file_content):
    data_names = file_content['机架号']
    unique_values = data_names.unique()
    if len(unique_values) == 2:
        data_name = 'FM+RM'
    else:
        data_name = unique_values[0]
    new_df = file_content[['轧辊号', '装入时刻', '结束时刻', '在线磨损']]
    name_to_en_mapping = {
        '轧辊号': 'roller_number',
        '装入时刻': 'start_time',
        '结束时刻': 'finish_time',
        '在线磨损': 'online_abrasion_loss'
    }
    new_df = new_df.rename(columns=name_to_en_mapping)
    return new_df, data_name


# 独热编码_对于轧辊号
def one_hot_Roller(roller_number):
    roller = {'BC01': 0, 'BC02': 0, 'BJ01': 0, 'BJ02': 0, 'BJ03': 0,
              'BJ04': 0, 'BJ05': 0, 'BJ06': 0, 'BJ07': 0, 'BJ08': 0}
    if roller_number in roller:
        roller[roller_number] = 1
        return roller
    else:
        return 0


# 独热编码_对于机架号
def one_hot_MillStand(millStand):
    millStands = {'FM': 0, 'RM': 0}
    if millStand in millStands:
        millStands[millStand] = 1
        return millStands
    else:
        return 0


# 将数据处理为模型所需要的数据
def data_to_model(duration, last_cutting_loss, last_rolling_wear,  rolling_weight, roller_number, millStand_number):
    # 独热编码特征
    roller = one_hot_Roller(roller_number)
    mill_stand = one_hot_MillStand(millStand_number)
    if roller == 0:
        return fail(msg="Error: Invalid roller number")
        # return 'Invalid roller number'
    if mill_stand == 0:
        return fail(msg="Error: Invalid Mill Stand number")
        # return 'Invalid MillStand'
    roller = list(roller.values())
    mill_stand = list(mill_stand.values())
    my_list = [duration] + [last_cutting_loss] + [last_rolling_wear] + [rolling_weight] + roller + mill_stand
    new_list = []
    # 遍历原始列表
    for item in my_list:
        try:
            # 尝试将元素转换为浮点数
            new_list.append(float(item))
        except ValueError:
            # 如果元素不能转换为浮点数（因为它实际上是一个字符串），那么就保持原样
            new_list.append(item)
    features = np.array(new_list).reshape(1, -1)
    return features
