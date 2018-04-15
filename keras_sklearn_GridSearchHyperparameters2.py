import numpy
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm

def create_keras_model(neurons=1,optimizer='adam',activation='relu',init_mode='uniform',dropout_rate=0.0,weight_constraint=0):
    model=Sequential()
    model.add(Dense(neurons,input_dim=8,kernel_initializer=init_mode,activation=activation,kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,kernel_initializer=init_mode,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    # print("neurons=",neurons,"||optimizer=",optimizer,"||activation=",activation,"||init_mode=",init_mode,
    #       "||dropout_rate=",dropout_rate,"||weight_constraint=",weight_constraint)
    return model
seed=7
numpy.random.seed(seed)
dataset=numpy.loadtxt("pima-indians-diabetes.data.csv",delimiter=',')
X=dataset[0:100,0:8]
Y=dataset[0:100:,8]


model=KerasClassifier(build_fn=create_keras_model,verbose=2)

batch_size=[100]
epochs=[1]
# batch_size=[80,100]
# epochs=[1,2]
neurons= [20,30]
optimizer = ['Adam']
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform']
# init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['relu', 'tanh']
# activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [1, 2]
# weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.3, 0.4]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#todo

param_grid=dict(batch_size=batch_size,epochs=epochs,neurons=neurons,optimizer=optimizer,init_mode=init_mode,activation=activation,weight_constraint=weight_constraint,dropout_rate=dropout_rate)
# {'activation': 'relu', 'batch_size': 100, 'dropout_rate': 0.4, 'epochs': 1, 'init_mode': 'uniform', 'neurons': 10, 'optimizer': 'Adam', 'weight_constraint': 2}
grid=GridSearchCV(estimator=model,cv=2,param_grid=param_grid,n_jobs=1,verbose=2)

from keras.callbacks import ModelCheckpoint
#Create instance of ModelCheckpoint
chk = ModelCheckpoint("myModel.h5", monitor='val_loss', save_best_only=False)
#add that callback to the list of callbacks to pass
callbacks_list = [chk]
#fit your model with your data. Pass the callback(s) here
grid_result =grid.fit(X,Y, callbacks=callbacks_list)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
