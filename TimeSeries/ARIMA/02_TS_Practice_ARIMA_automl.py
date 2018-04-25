import warnings
from pandas import read_csv
from pandas import datetime
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order,X_train_ration=0.66):
	# prepare training dataset
	train_size = int(len(X) * X_train_ration)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load dataset
data_file1="D:\pythonProject\LSTM\LstmPm25\shampoo.csv"
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
series1 = read_csv(data_file1, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]
d_values = range(0, 3)
q_values = range(0, 3)
X_train_ration=0.8
warnings.filterwarnings("ignore")
# evaluate_models(series1.values, p_values, d_values, q_values)
data_file2="D:\pythonProject\LSTM\daily-total-female-births-in-cal.csv"
series2 = Series.from_csv(data_file2, header=0)
evaluate_models(series1.values, p_values, d_values, q_values)

