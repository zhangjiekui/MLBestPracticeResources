#How to Make Out-of-Sample Forecasts with ARIMA in Python
#https://machinelearningmastery.com/make-sample-forecasts-arima-python/



from pandas import Series
from matplotlib import pyplot


def load_explore_split_data():
    data_file2 = "D:\pythonProject\LSTM\daily-minimum-temperatures-in-me.csv"
    series = Series.from_csv(data_file2, header=0)
    print(series.head(20))
    series.plot()
    pyplot.show()
    split_point=len(series)-7
    traindata,valdata=series[0:split_point],series[split_point:]
    print("train_data %d,validation_data %d" %(len(traindata),len(valdata)))
    traindata.to_csv("traindata.csv")
    valdata.to_csv("valdata.csv")


# load_explore_split_data() #todo

from statsmodels.tsa.arima_model import ARIMA
import numpy as np

#create a differenced series
def difference(dataset,interval=1):
    diff=list()
    for i in range(interval,len(dataset)):
        value=dataset[i]-dataset[i-interval]
        diff.append(value)
    return np.array(diff)
# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]


series=Series.from_csv("traindata.csv",header=None)
X=series.values
days_in_year=365
differenced=difference(X,days_in_year)
print("differenced.len",len(differenced))
series=Series(differenced)
series.plot()
pyplot.show()

# fit model
model = ARIMA(differenced, order=(7,0,1))
model_fit = model.fit(disp=0)
# print summary of fit model
print(model_fit.summary())
# one-step out-of sample forecast
forecast = model_fit.forecast()[0]
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)

#The predict function can be used to predict arbitrary in-sample and out-of-sample time steps,
# including the next out-of-sample forecast time step.
#The predict function requires a start and an end to be specified,
# these can be the indexes of the time steps relative to the beginning of the training data
# used to fit the model
start_index = len(differenced)
end_index = len(differenced)
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
forecast = inverse_difference(X, forecast, days_in_year)
print('Forecast: %f' % forecast)

# start_index = '1990-12-25'
# end_index = '1990-12-25'
# forecast = model_fit.predict(start=start_index, end=end_index)
#
#
# # from pandas import datetime
# # start_index = datetime(1990,12,25)
# # end_index = datetime(1990,12,26).date()
# # forecast = model_fit.predict(start=start_index, end=end_index)
# forecast = inverse_difference(X, forecast, days_in_year)
# print('Forecast: %f' % forecast)

# multi-step out-of-sample forecast
forecast = model_fit.forecast(steps=7)[0]
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1


# multi-step out-of-sample forecast
start_index = len(differenced)
end_index = start_index + 6
forecast = model_fit.predict(start=start_index, end=end_index)
# invert the differenced forecast to something usable
history = [x for x in X]
day = 1
for yhat in forecast:
	inverted = inverse_difference(history, yhat, days_in_year)
	print('Day %d: %f' % (day, inverted))
	history.append(inverted)
	day += 1
