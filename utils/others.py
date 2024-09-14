import joblib
import numpy as np
from flask import Flask, request
from utils.data_processing import data_to_gantt, one_hot_Roller, one_hot_MillStand
from utils.response import success, fail


def getModelResult():
    data = request.get_json()
    duration = data["rollingTime"]
    last_cutting_loss = data["lastCutting"]
    last_rolling_wear = data["lastRollingWear"]
    rolling_weight = data["rollingWeight"]
    roller_number = data["roller"]
    millStand_number = data['millStand']
    # 独热编码特征
    roller = one_hot_Roller(roller_number)
    mill_stand = one_hot_MillStand(millStand_number)
    if roller == 0:
        return fail(msg="Error: Invalid roller number")
    if mill_stand == 0:
        return fail(msg="Error: Invalid MillStand")
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
    scaler_x = joblib.load('./models/scaler_x_train.pkl')
    scaler_y = joblib.load('./models/scaler_y_train.pkl')
    _std_x_test = scaler_x.transform(features)
    print(_std_x_test)
    svr_model = joblib.load('./models/SVR_1thCV.dat')
    gbr_model = joblib.load('./models/GBR_1thCV.dat')
    gpr_model = joblib.load('./models/GPR_1thCV.dat')
    rfr_model = joblib.load('./models/RFR_1thCV.dat')
    ridge_model = joblib.load('./models/Ridge_1thCV.dat')
    svr_prediction = svr_model.predict(_std_x_test)
    print(svr_prediction)
    gbr_prediction = gbr_model.predict(_std_x_test)
    gpr_prediction = gpr_model.predict(_std_x_test)
    rfr_prediction = rfr_model.predict(_std_x_test)
    ridge_prediction = ridge_model.predict(_std_x_test)
    svr_prediction = scaler_y.inverse_transform(svr_prediction.reshape(-1, 1))
    gbr_prediction = scaler_y.inverse_transform(gbr_prediction.reshape(-1, 1))
    gpr_prediction = scaler_y.inverse_transform(gpr_prediction.reshape(-1, 1))
    rfr_prediction = scaler_y.inverse_transform(rfr_prediction.reshape(-1, 1))
    ridge_prediction = scaler_y.inverse_transform(ridge_prediction.reshape(-1, 1))
    result = {
        "svr": svr_prediction[0][0],
        "gbr": gbr_prediction[0][0],
        "gpr": gpr_prediction[0][0],
        "rfr": rfr_prediction[0][0],
        "ridge": ridge_prediction[0][0],
    }
    return success(msg="请求成功", data=result)