
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import set_printoptions
import matplotlib.pyplot as plt
# 设置显示中文字体
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置显示正常符号
mpl.rcParams["axes.unicode_minus"] = False
pd.set_option('display.unicode.east_asian_width',True)

# 设置列对齐
pd.set_option('display.unicode.ambiguous_as_wide',True)
pd.set_option('display.unicode.east_asian_width',True)

test= pd.read_excel(io="番剧数据-原始数据【2015-2023】.xlsx",sheet_name="Sheet1")
test1 = pd.read_excel(io="番剧数据-原始数据【2015-2023-不同地区】.xlsx",sheet_name="Sheet1")
def clean(test):
    print("============原始前五行数据")
    print(test.head())

    for x in range(0,test.shape[0]):
        if ("万" in test['系列追番数'].iloc[x]):
            df = test['系列追番数'].iloc[x].replace('万系列追番', '').replace('万追番', '').replace('万追剧', '').replace('万系列追剧', '')
            test.loc[x,'系列追番数'] = float(df)*100000000
        else:
            test.loc[x,'系列追番数'] = int(test.loc[x,'系列追番数'].replace('追剧', '').replace('系列追番', '').replace('追番', '').replace('追剧', '').replace('系列追剧', ''))


    # 删掉名称一列
    test.drop(["名称"],axis=1,inplace=True)
    print("\n"+"============数值处理之后数据")
    print(test.head())

    # 输出数据清洗前的数据信息
    print("\n"+"============数据清洗前的数据信息")
    test.info()



    #========缺失值处理
    # 将0替换成Nan
    test.replace(to_replace=0,value=np.nan,inplace=True)
    def missvalue_test(data, column, method="Delete"):
       if (np.all(pd.notnull(data))):
           print(column+"无缺失值")
           # return data
       else:
           if method == "Delete":
               print('全部删除'+column+'列缺失值所在行...')
               print('=' * 70)
               data.dropna(how = "any",inplace=True)
               # return data
           if method == 'fill':
               print('填补'+column+'列缺失值为列平均值...')
               data[column].fillna(data[column].mean(), inplace=True)
               # return data

    # 将评分列的缺失值替换成该列的平均值
    missvalue_test(test,"评分","fill")
    # 删除播放量，系列追番数，转发量中的缺失值
    missvalue_test(test,"播放量","Delete")
    missvalue_test(test,"系列追番数","Delete")
    missvalue_test(test,"转发量","Delete")
    missvalue_test(test,"投币量","Delete")
    missvalue_test(test,"点赞量","Delete")
    print("\n"+"============缺失值处理后")
    test.info()


    # 可视化看数据分布，这里是pandas模块里的画图
    test.plot(kind='box',subplots=True,layout=(3,3),sharex=False,figsize = (10,10))
    plt.savefig("【箱线图】2015-2023年播放量情况-不同地区")
    plt.show()


    # 异常值处理
    #  异常值检验函数 =========================
    def outlier_test(data, column):
           print(f'以 {column} 列为依据，使用 上下截断点法(iqr) 检测异常值...')
           print('=' * 70)
           # 四分位点；这里调用函数会存在异常
           column_iqr = np.quantile(data[column], 0.75) - np.quantile(data[column], 0.25)
           # 1，3 分位数
           (q1, q3) = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)
           # 计算上下截断点
           upper, lower = (q3 + 1.5 * column_iqr), (q1 - 1.5 * column_iqr)
           # 检测异常值
           outlier = data[(data[column] <= lower) | (data[column] >= upper)]
           print(f'第一分位数: {q1}, 第三分位数：{q3}, 四分位极差：{column_iqr}')
           print(f"上截断点：{upper}, 下截断点：{lower}")
           return outlier, upper, lower
    # 对系列追番数，采取四分位法检测异常值
    outlier, upper, lower = outlier_test(data=test, column="系列追番数")

    # 对系列追番数
    # outlier1, upper1, lower1 = outlier_test(data=test, column="系列追番数")
    # outlier = pd.concat([outlier,outlier1],axis=0)

    # 输出异常值信息
    print("\n"+"=====================异常值信息")
    outlier.info()

    # 删除异常值
    test.drop(index=outlier.index, inplace=True)

    # 输出删除异常值后表格信息
    print("\n"+"=====================删除异常值后表格信息")
    test.info()
    return test


test = clean(test)
test1 = clean(test1)
# 保存清洗后数据
test.to_excel("番剧数据-清洗后数据【2015-2023】.xlsx")
test1.to_excel("番剧数据-清洗后数据【2015-2023-不同地区】.xlsx")


# ====================对表格  番剧数据-清洗后数据【2015-2023-不同地区】的处理
test1.drop(["所属类别"],axis=1,inplace=True)
test1=test1.groupby(by = ['年份',"地区"]).mean()
# 归一化
transformer = MinMaxScaler(feature_range=(0,100))
newtest1 = transformer.fit_transform(test1)
set_printoptions(precision=3)   #小数点后3位
# 输出归一化后列表
newtest1=pd.DataFrame(newtest1,index=test1.index,columns=test1.columns)
newtest1["受欢迎程度"]=newtest1["播放量"]*100+newtest1["评分"]*90+newtest1["点赞量"]*85+newtest1["投币量"]*85+newtest1["转发量"]*55+newtest1["弹幕数"]*35+newtest1["系列追番数"]*88
newtest1.to_excel("番剧数据-归一化后数据【2015-2023-不同地区】.xlsx")
print("----------------")
print(newtest1)


# ================================对表格 番剧数据-清洗后数据【2015-2023】 的处理
# 分类,1恋爱，2冒险，3搞笑，4治愈，5热血
data = pd.DataFrame(columns=test.columns)
for x in range(0,test.shape[0]):
  if("恋爱" in  test['所属类别'].iloc[x]):
      data.loc[len(data)] = test.iloc[x]
      data.loc[len(data)-1,'所属类别'] = "恋爱"
  elif("冒险" in  test['所属类别'].iloc[x]):
      data.loc[len(data)] = test.iloc[x]
      data.loc[len(data)-1,'所属类别'] = "冒险"
  elif("搞笑" in  test['所属类别'].iloc[x]):
      data.loc[len(data)] = test.iloc[x]
      data.loc[len(data)-1,'所属类别'] = "搞笑"
  elif("治愈" in  test['所属类别'].iloc[x]):
      data.loc[len(data)] = test.iloc[x]
      data.loc[len(data)-1,'所属类别'] = "治愈"
  elif ("热血" in test['所属类别'].iloc[x]):
      data.loc[len(data)] = test.iloc[x]
      data.loc[len(data)-1,'所属类别'] = "热血"
data=data.groupby(by = ['年份',"所属类别"]).mean()
print(data)


# 归一化
transformer = MinMaxScaler(feature_range=(0,100))
newtest = transformer.fit_transform(data)
set_printoptions(precision=3)   #小数点后3位
# 输出归一化后列表
newtest=pd.DataFrame(newtest,index=data.index,columns=data.columns)
print(newtest)
newtest.to_excel("番剧数据-归一化后数据【2015-2023】（无受欢迎程度，有分类）.xlsx")





# #======================================归一化
# test.drop(["所属类别"],axis=1,inplace=True)
# test=test.groupby(by = '年份').mean()
# # 归一化
# transformer = MinMaxScaler(feature_range=(0,100))
# newtest1 = transformer.fit_transform(test)
# set_printoptions(precision=3)   #小数点后3位
# # 输出归一化后列表
# newtest1=pd.DataFrame(newtest1,index=test.index,columns=test.columns)
# newtest1["受欢迎程度"]=newtest1["播放量"]*100+newtest1["评分"]*90+newtest1["点赞量"]*85+newtest1["投币量"]*85+newtest1["转发量"]*55+newtest1["弹幕数"]*35+newtest1["系列追番数"]*88
# print(newtest1)
# newtest1.to_excel("番剧数据-归一化后数据【2015-2023】（未分类）.xlsx")