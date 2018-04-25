# [python] 时间序列分析之ARIMA
# https://blog.csdn.net/hal_sakai/article/details/51965657

import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422,
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355,
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767,
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232,
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248,
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722,
11999,9390,13481,14795,15845,15271,14686,11054,10395]

dta=pd.Series(dta)
dta.index = pd.Index(sm.tsa.datetools.dates_from_range('2011','2100'))
dta.plot(figsize=(12,8))
plt.show()

#
# # ARIMA 模型对时间序列的要求是平稳型。
# # 因此，当你得到一个非平稳的时间序列时，首先要做的即是做时间序列的差分，直到得到一个平稳时间序列。
# # 如果你对时间序列做d次差分才能得到一个平稳序列，那么可以使用ARIMA(p,d,q)模型，其中d是差分次数。
# # .....差分，寻找合适的d参数
# fig = plt.figure(figsize=(12,8))
# ax1= fig.add_subplot(111)
# diff1 = dta.diff(1) #一阶差分 d=1
# diff1.plot(ax=ax1)
# plt.show()
#
# fig = plt.figure(figsize=(12,8))
# ax2= fig.add_subplot(111)
# diff2 = dta.diff(2) #二阶差分 d=2
# diff2.plot(ax=ax2)
# plt.show()

# 现在我们已经得到一个平稳的时间序列，接来下就是选择合适的ARIMA模型，即ARIMA模型中合适的p,q。
# 第一步我们要先检查平稳时间序列的自相关图和偏自相关图。

diff1= dta.diff(1).dropna()#我们已经知道要使用一阶差分的时间序列，之前判断差分的程序可以注释掉
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta,lags=40,ax=ax2)
plt.show()

from statsmodels.tsa.stattools import adfuller as ADF
print(u'原始序列的ADF检验结果为：', ADF(dta))
# 原始序列的ADF检验结果为： (-1.1636399963151427, 0.689038130711985, 12, 77, {'1%': -3.518281134660583, '5%': -2.899878185191432, '10%': -2.5872229937594873}, 1364.8131984050463)
#返回值依次为                 adf、pvalue(p值显著大于0.05，该序列为非平稳序列)、usedlag、nobs、critical values、icbest、regresults、resstore
print(u'diff1序列的ADF检验结果为：', ADF(diff1))

#白噪声检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'diff1序列的白噪声检验结果为：', acorr_ljungbox(diff1,lags=1))
# diff1序列的白噪声检验结果为： (array([2.4989039]), array([0.11392556]))
# P值小于0.05，所以一阶差分后的序列为平稳非白噪声序列,本处 array([0.11392556]) ，还不行
print(u'diff1序列的白噪声检验结果为——lags=40：', acorr_ljungbox(diff1,lags=40))
