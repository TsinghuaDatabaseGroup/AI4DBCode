from pyecharts.charts import Pie  # 饼图所导入的包
from pyecharts import options as opts  # 全局设置所导入的包


def Pie1():
    pie = (
        Pie()
        .add("", [['Nest Queries', '257'],
                  ['Non-Nest Queries', '228'],
                  ])  # 加入数据
        .set_global_opts(title_opts=opts.TitleOpts(title="Nest/Non-Nest Query Distribution"),
                         legend_opts=opts.LegendOpts(pos_left=160, pos_top=100))  # 全局设置项
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%", )))  # 样式设置项
    return pie


Pie1().render("/Users/zhanglixi/Desktop/draw-code/test.html")  # 保存图片为HTML网页




