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
print(test.head())


# # 将“播放量”列中的“万，亿”去掉，再转换成浮点数乘一万，一亿
# for x in range(0,test.shape[0]):
#     if('万' in test['播放量'].iloc[x]):
#         df = (test['播放量'].iloc[x]).replace('万', '')
#         test.iloc[x,6] = float(df)*10000
#     else:
#         df = test['播放量'].iloc[x].replace('亿', '')
#         test.iloc[x,6] = float(df)*100000000


print(type(test))
for x in range(0,test.shape[0]):
    if ("万" in test['系列追番数'].iloc[x]):
        df = test['系列追番数'].iloc[x].replace('万系列追番', '').replace('万追番', '').replace('万追剧', '').replace('万系列追剧', '')
        test.iloc[x,8] = float(df)*100000000
    else:
        test.iloc[x,8] = int(test.iloc[x,8].replace('追剧', '').replace('系列追番', '').replace('追番', '').replace('追剧', '').replace('系列追剧', ''))


# 删掉名称一列
test.drop(["名称"],axis=1,inplace=True)
print(test.head())
# 保存为新的xlsx表格
test.to_excel('新表.xlsx', index=False)
# 输出数据清洗前的数据信息
print("============数据清洗前的数据信息")
test.info()



#========缺失值处理
# 将0替换成Nan
test.replace(to_replace=0,value=np.nan,inplace=True)
def missvalue_test(data, column, method="Delete"):
   if (np.all(pd.notnull(data))):
       print("无缺失值")
       # return data
   else:
       if method == "Delete":
           print('全部删除缺失值所在行...')
           print('=' * 70)
           data.dropna(how = "any",inplace=True)
           # return data
       if method == 'fill':
           print('填补缺失值为列平均值...')
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
print("============缺失值处理后")
test.info()


# 可视化看数据分布，这里是pandas模块里的画图
# test.plot(kind='density',subplots=True,layout=(3,3),sharex=False,figsize = (10,10))
# plt.savefig("密度图")
# test.hist(figsize = (10,10))
# plt.savefig("柱状图")
test.plot(kind='box',subplots=True,layout=(3,3),sharex=False,figsize = (10,10))
plt.savefig("箱线图")
plt.show()

#==============这里原本想做一个简单函数变换更符合正态分布。。。。感觉出来的结果不太合适
# for i in range(1, len(test["评分"].tolist())):
#     test["评分"].iloc[i] = math.log(test["评分"].iloc[i],1.5)
# for i in range(1, len(test["播放量"].tolist())):
#     test["播放量"].iloc[i] = math.log(test["评分"].iloc[i],0.5)
# for i in range(1, len(test["转发量"].tolist())):
#     test["转发量"].iloc[i] = math.log(test["评分"].iloc[i],0.5)
# test.plot(kind='density',subplots=True,layout=(3,3),sharex=False,figsize = (10,10))
# plt.savefig("密度图1")
# plt.show()

# 以下方法检验数据是否符合正态分布，，，，评分列不符合，另一方面从可视化图像可以看出不符合，故不能用3sigma方法检验异常值
# # .kstest方法：KS检验，参数分别是：待检验的数据，检验方法（这里设置成norm正态分布），均值与标准差
# # 结果返回两个值：statistic → D值，pvalue → P值
# # p值大于0.05，为正态分布
# from scipy import stats
# mean = test["评分"].mean()	#===均值
# std = test["评分"].std()		#====标准差
# print(stats.kstest(test["评分"],'norm',(mean,std)))

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
# 对评分，采取四分位法检测异常值
outlier, upper, lower = outlier_test(data=test, column="评分")

# 对系列追番数
outlier1, upper1, lower1 = outlier_test(data=test, column="系列追番数")
outlier = pd.concat([outlier,outlier1],axis=0)

# 输出异常值信息
print("=====================异常值信息")
outlier.info()

# 删除异常值
test.drop(index=outlier.index, inplace=True)

# 输出删除异常值后表格信息
print("=====================删除异常值后表格信息")
test.info()

#保存清洗后数据
test.to_excel("番剧数据-清洗后数据【2015-2023】.xlsx")

# 分类读取
LianAi = pd.DataFrame(columns=test.columns)
MaoXian = pd.DataFrame(columns=test.columns)
GaoXiao = pd.DataFrame(columns=test.columns)
ZhiYu = pd.DataFrame(columns=test.columns)
ReXue = pd.DataFrame(columns=test.columns)
for x in range(0,test.shape[0]):
    if("恋爱" in  test['所属类别'].iloc[x]):
        LianAi.loc[len(LianAi)]=(test.iloc[x])

    elif("冒险" in  test['所属类别'].iloc[x]):
        MaoXian.loc[len(MaoXian)]=(test.iloc[x])
    elif("搞笑" in  test['所属类别'].iloc[x]):
        GaoXiao.loc[len(GaoXiao)]=(test.iloc[x])
    elif("治愈" in  test['所属类别'].iloc[x]):
        ZhiYu.loc[len(ZhiYu)]=(test.iloc[x])
    elif ("热血" in test['所属类别'].iloc[x]):
        ReXue.loc[len(ReXue)] = (test.iloc[x])

GaoXiao.drop(["所属类别"],axis=1,inplace=True)


# # # 归一化
# transformer = MinMaxScaler(feature_range=(0,1))
# newtest = transformer.fit_transform(GaoXiao)
# set_printoptions(precision=3)   #小数点后3位
# # 输出归一化后列表
# print(newtest)

# 以搞笑类为例===========
# 数据集划分
x = GaoXiao[["年份","点赞量","投币量","转发量","弹幕数","评分","系列追番数"]]
# 此处不转换成数组就是Dataframe对象
# x = np.array(x)
y =GaoXiao["播放量"]
# y = np.array(y)
print(x[:5])
print(y[:5])
#按1：9划分训练集与测试集
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.10,random_state=42)
print("X_train",X_train)
print("X_test",X_test)
print("Y_train",Y_train)
print("Y_test",Y_test)



# lr = linear_model.LinearRegression()
# lr.fit(x,y)
# test1 = np.array([[1009.0,20190903.0],
#                  [1007.0,20190927.0]])
# for item in test1:
#     item1 = copy.deepcopy(item)
#     print(item,":",lr.predict(item1).reshape(1,-1))





# # 归一化
# transformer = MinMaxScaler(feature_range=(0,1))
# newtest = transformer.fit_transform(test)
# set_printoptions(precision=3)   #小数点后3位
# # 输出归一化后列表
# print(newtest)
