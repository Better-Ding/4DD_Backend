import base64
from io import BytesIO
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def color(row):
    _c_dict = {'BJ07': '#E64646', 'BJ01': '#E69646', 'BJ02': '#34D05C',
               'BC02': '#34D0C3', 'BC01': '#3475D0', 'BJ03': '#A0522D',
               'BJ04': '#FF1493', 'BJ08': '#800080', 'BJ05': '#808080',
               'BJ06': '#FFFF00'}
    return _c_dict[row['roller_number']]


def getGantt(data, data_name):
    # df = pd.read_csv('./FM-roller-picture.csv')
    # pic_name = 'FM'
    df = data
    pic_name = data_name
    # 数据预处理
    df['start_time'] = pd.to_datetime(df.start_time)
    df['finish_time'] = pd.to_datetime(df.finish_time)
    # project start date
    proj_start = df.start_time.min()
    # number of days from project start to task start, left参数
    df['start_num'] = (df.start_time - proj_start).dt.days
    # number of days from project start to end of tasks
    df['end_num'] = (df.finish_time - proj_start).dt.days
    # days between start and end of each task,width参数
    df['days_start_to_end'] = df.end_num - df.start_num

    # #E64646:红色,#E69646:黄棕色，#34D05C：绿色，#34D0C3：天蓝，#3475D0：蓝色
    # #A0522D：棕色，#FF1493：粉红色, #800080:紫色；#808080：灰色,'#FFFF00':黄色
    df['color'] = df.apply(color, axis=1)

    fig, ax = plt.subplots(1, figsize=(16, 10))
    ax.barh(y=df.online_abrasion_loss
            , width=df.days_start_to_end
            , left=df.start_num
            , height=0.1
            , color=df.color
            )
    # 设置不同种类的轧辊型号
    if pic_name == 'FM':
        c_dict = {'BJ07': '#E64646', 'BJ01': '#E69646', 'BJ02': '#34D05C',
                  'BC02': '#34D0C3', 'BC01': '#3475D0', 'BJ03': '#A0522D',
                  'BJ04': '#FF1493'}
        legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
        plt.legend(handles=legend_elements)
    elif pic_name == 'RM':
        c_dict = {'BJ02': '#34D05C', 'BJ04': '#FF1493',
                  'BJ08': '#800080', 'BJ05': '#808080', 'BJ06': '#FFFF00'}
        legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
        plt.legend(handles=legend_elements)
    elif pic_name == 'FM+RM':
        c_dict = {'BJ07': '#E64646', 'BJ01': '#E69646', 'BJ02': '#34D05C',
                  'BC02': '#34D0C3', 'BC01': '#3475D0', 'BJ03': '#A0522D',
                  'BJ04': '#FF1493', 'BJ08': '#800080', 'BJ05': '#808080',
                  'BJ06': '#FFFF00'}
        legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
        plt.legend(handles=legend_elements)

    # legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
    # plt.legend(handles=legend_elements)
    # 设置x轴刻度和刻度标签
    xticks = np.arange(0, df.end_num.max() + 1, 60)
    xticks_labels = pd.date_range(proj_start,
                                  end=df.finish_time.max()).strftime("%Y/%m/%d")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels[::60], rotation=60, fontsize=10)

    # 画垂直于x轴的虚线
    x_set = np.concatenate((df.start_num, df.end_num))
    y_set = np.concatenate((df.online_abrasion_loss, df.online_abrasion_loss))

    # 画左侧垂线线段
    ymin_start = []
    ymax_start = []
    # print(df.start_num.unique())
    # print(df.end_num.unique().shape[0])
    for i in range(df.start_num.unique().shape[0]):
        ymin_start.append(df.loc[2 * i, 'online_abrasion_loss'])
        ymax_start.append(df.loc[2 * i + 1, 'online_abrasion_loss'])
    ax.vlines(x=df.start_num.unique(), ymin=ymin_start,
              ymax=ymax_start, linestyles='dashed', colors='k')
    # # 画右侧垂线线段
    ymin_end = []
    ymax_end = []
    for i in range(df.end_num.unique().shape[0]):
        ymin_end.append(df.loc[2 * i, 'online_abrasion_loss'])
        ymax_end.append(df.loc[2 * i + 1, 'online_abrasion_loss'])
    ax.vlines(x=df.end_num.unique(), ymin=ymin_end,
              ymax=ymax_end, linestyles='dashed', colors='k')

    ax.set_title("Gantt chart of %s" % pic_name, fontsize=18)
    ax.set_xlabel("Timeline", fontsize=16)
    ax.set_ylabel("Rolling wear loss", fontsize=16)
    # 转成图片的步骤
    sio = BytesIO()
    plt.savefig(sio, format="png")
    data = base64.b64encode(sio.getvalue()).decode()
    plot_url = 'data:image/png;base64,' + data
    plt.close()
    return plot_url
