import pickle
from io import StringIO
import chardet
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request
from flask_cors import CORS
from utils.Roller_Gantt import getGantt
from utils.data_processing import data_to_gantt, data_to_model
from utils.load_model import batch_train
from utils.response import success, fail

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# CORS(app, resources={r"/*": {"origins": "http://114.55.87.45:2423"}})
scaler_x = joblib.load('./models/scaler_x_train.pkl')
scaler_y = joblib.load('./models/scaler_y_train.pkl')
svr_model = joblib.load('./models/SVR_1thCV.dat')
gbr_model = joblib.load('./models/GBR_1thCV.dat')
gpr_model = joblib.load('./models/GPR_1thCV.dat')
rfr_model = joblib.load('./models/RFR_1thCV.dat')


@app.route('/getGantt', methods=["Post"])
def obtainGantt():
    file = request.files['file']
    # 先读取一部分字节来检测编码
    rawdata = file.stream.read(1024)
    result = chardet.detect(rawdata)
    file.stream.seek(0)  # 重置文件指针至文件头，以便全部读取
    file_content = file.stream.read().decode(result['encoding'])
    file = pd.read_csv(StringIO(file_content))
    if file.isnull().values.any():
        return fail("数据中存在空缺值")
    # 获得机器学习预测结果
    res = batch_train(file, scaler_x, scaler_y, svr_model, gbr_model, gpr_model, rfr_model)
    print(res)
    if type(res) == list:
        # 输入过来的数据加以处理，输出甘特图
        data, data_name = data_to_gantt(file)
        # 输入数据以绘制gantt图
        plot_url = getGantt(data=data, data_name=data_name)
        res = {
            "plot_url": plot_url,
            "predict_res": res,
        }
        return success(msg="上传成功", data=res)
    else:
        return res


@app.route('/getModelRes', methods=["Post"])
def getModelResult():
    data = request.get_json()
    if "" in data.values():
        return fail("上传数据中存在空值")
    duration = data["rollingTime"]
    last_cutting_loss = data["lastCutting"]
    last_rolling_wear = data["lastRollingWear"]
    rolling_weight = data["rollingWeight"]
    roller_number = data["roller"]
    millStand_number = data['millStand']
    features = data_to_model(duration, last_cutting_loss, last_rolling_wear, rolling_weight, roller_number, millStand_number)
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
            "rollingTime": duration,
            "rollingWeight": rolling_weight,
            "predict_res": svr_prediction.round(3),
            "residuals": residuals.round(2)
        }
        return success(msg="请求成功", data=result)
    else:
        return features


@app.route('/hello', methods=["Get"])
def helloWorld():
    return success(msg="hello world")


if __name__ == '__main__':
    app.run()
