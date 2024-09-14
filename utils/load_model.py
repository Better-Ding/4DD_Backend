import joblib
import numpy as np
import pandas as pd
from utils.data_processing import detect_encoding, data_to_gantt, one_hot_Roller, one_hot_MillStand, data_to_model


# 训练SVR模型，绘制甘特图
def batch_train(file, scaler_x, scaler_y, svr_model, gbr_model, gpr_model, rfr_model):
    df = file
    # 将运行时间改为天
    df['装入时刻'] = pd.to_datetime(df['装入时刻'])
    df['结束时刻'] = pd.to_datetime(df['结束时刻'])
    df['持续时间_天'] = ((df['结束时刻'] - df['装入时刻']).dt.total_seconds() / (24 * 60 * 60))
    # 按照日期对DataFrame进行排序,防止用户未按时间顺序填入
    df_sorted = df.sort_values('装入时刻')
    data_names = df_sorted['轧辊号']
    # 获取轧辊所有种类
    roller_types = data_names.unique()
    data_feature = pd.DataFrame()
    # 创建两个新的特征，上次人工切削量和上次在线磨损量
    for roller in roller_types:
        roller_data = df_sorted[df_sorted['轧辊号'] == roller].copy()
        roller_data['上次人工切削量'] = roller_data['磨削量'].shift(fill_value=0)
        roller_data['上次在线磨损量'] = roller_data['在线磨损'].shift(fill_value=0)
        data_feature = pd.concat([data_feature, roller_data])
    # 记录预测结果列表
    res_list = []
    for index, data in data_feature.iterrows():
        duration = data["持续时间_天"]
        last_cutting_loss = data["上次人工切削量"]
        last_rolling_wear = data["上次在线磨损量"]
        rolling_weight = data["轧制重量（t）"]
        roller_number = data["轧辊号"]
        millStand_number = data['机架号']
        features = data_to_model(duration, last_cutting_loss, last_rolling_wear,  rolling_weight, roller_number, millStand_number)
        if type(features) == np.ndarray:
            _std_x_test = scaler_x.transform(features)
            svr_prediction = svr_model.predict(_std_x_test)
            gbr_prediction = gbr_model.predict(_std_x_test)
            gpr_prediction = gpr_model.predict(_std_x_test)
            rfr_prediction = rfr_model.predict(_std_x_test)
            svr_prediction = scaler_y.inverse_transform(svr_prediction.reshape(-1, 1))[0][0]
            gbr_prediction = scaler_y.inverse_transform(gbr_prediction.reshape(-1, 1))[0][0]
            gpr_prediction = scaler_y.inverse_transform(gpr_prediction.reshape(-1, 1))[0][0]
            rfr_prediction = scaler_y.inverse_transform(rfr_prediction.reshape(-1, 1))[0][0]
            predict_mean = (svr_prediction + gbr_prediction + gpr_prediction + rfr_prediction) / 4
            residuals = abs(svr_prediction - predict_mean)
            result = {
                "millStand": millStand_number,
                "roller": roller_number,
                "rollingTime": round(duration,3),
                "rollingWeight": rolling_weight,
                "predict_res": svr_prediction.round(3),
                "residuals": residuals.round(2)
            }
            res_list.append(result)
        else:
            return features
    return res_list

