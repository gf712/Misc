#!/usr/bin/env python

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from time import time

def main(train_path, test_path, n_estimators, estimator_params, scoring, n_folds):

    ## load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    

    ## concatenate data to make sure all features are used in label encoding
    
    traintest = pd.concat((train.ix[:,1:-1], test.ix[:,1:]),axis=0)
    cont = np.array([x.startswith('cont') for x in traintest.columns])
    sc=StandardScaler()
    traintest.ix[:, cont] = sc.fit_transform(traintest.ix[:,cont])
    traintestOHE = pd.get_dummies(traintest)
    testset = traintestOHE.iloc[len(train):,:]
    trainset = pd.concat((traintestOHE.iloc[:len(train),:],train.loss),axis=1)
    train_y = trainset.iloc[:,-1].squeeze()
    trainset = trainset.iloc[:,:-1]

    ## get IDs for submission
    ids = test.id

    ## remove useless pointers
    del train, test, traintestOHE, traintest

    ## some logging
    print('-'*60)
    print('Loaded data')
    print('Size of test set: {}'.format(len(testset)))
    print('Size of train set: {}'.format(len(trainset)))
    print('-'*60)

    ## instantiate variables
    prediction = np.zeros((len(testset),n_folds,n_estimators))
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=5,
                                  patience=5, verbose=0, mode='min')
    validation_loss = list()


    ## do the bagging kfold
    folds = KFold(n_splits = n_folds, shuffle = True)
    validation_score = np.zeros((n_folds))

    for i, (train_index, test_index) in enumerate(folds.split(trainset)):

        start_time = time()

        train_X_nfolds = trainset.ix[train_index,:]
        train_y_nfolds = train_y[train_index]

        ## get number of data points per fold
        estimator_size  = int(len(train_X_nfolds) / n_estimators)

        ## keep track of score per fold in validation set
        validation_X = trainset.ix[test_index,:]
        validation_y = train_y[test_index]
        validation_score_i = np.zeros((n_estimators))
        
        ## start the bagging 
        for j, x in enumerate(range(n_estimators)):

            ## print current iteration
            print('\nFold: {}, Estimator: {}'.format(i+1, j+1))
            print('Params: {}'.format([estimator_params['dropout_rate'][j],
                                       estimator_params['layer1'][j],
                                       estimator_params['layer2'][j],
                                       estimator_params['layer3'][j],
                                       estimator_params['layer4'][j]]))

            ## draw data randomly with replacement
            index = np.random.choice(len(train_X_nfolds), size=estimator_size*2, replace=True)
            train_Xi = train_X_nfolds.iloc[index,:]
            train_yi = train_y_nfolds.iloc[index]

            ## compile model
            model = KerasRegressor(build_fn=baseline_model, nb_epoch=50, 
                                   batch_size=35, verbose=0,
                                   dropout_rate=estimator_params['dropout_rate'][j],
                                   layer1=estimator_params['layer1'][j],
                                   layer2=estimator_params['layer2'][j],
                                   layer3=estimator_params['layer3'][j],
                                   layer4=estimator_params['layer4'][j])

            ## train model
            history = model.fit(train_Xi.as_matrix(), train_yi.as_matrix(),
                                validation_split=0.25, callbacks=[earlyStopping],
                                show_accuracy=True)

            ## update test prediction
            prediction[:,i,j] = model.predict(testset.as_matrix())
            validation_score_i[j] = mean_absolute_error(model.predict(validation_X.as_matrix()),
                                                      validation_y.as_matrix())
            # retain the validation values
            validation_loss.append(history.history['val_mean_absolute_error'])
            print('Results after {} epochs'.format(len(history.history['mean_absolute_error'])))
            print('\nTraining MAE: {}'.format(history.history['mean_absolute_error'][-1]))
            print('Callback Validation MAE: {}'.format(history.history['val_mean_absolute_error'][-1]))
            print('Validation Set MAE: {}\n'.format(validation_score_i[j]))
            print('-'*60)


        print('\nFold {} MAE: {}\n (std: {})'.format(i+1, validation_score_i.mean(),
                                                     validation_score_i.std()))
        print('Time: {} seconds'.format(time()-start_time))
        print('-'*60)
        print('-'*60)

        ## store the score of validation set
        validation_score[i] = validation_score_i.mean()

    ## store validation scores in numpy array
    # np.savetxt(fname='./BaggingKerasVal.npy',X=np.array(validation_loss),
    #            fmt='%10.5f')

    ## get the average prediction per test data point
    prediction = prediction.mean(axis=-1).mean(axis=-1)

    ## some stats logging
    print('Validation scores: {}'.format(validation_score))
    print('Validation score mean: {}'.format(validation_score.mean()))
    print('Validation score std: {}'.format(validation_score.std()))
    print('-'*30+'\n')
    print('Training score: {}'.format(mean_absolute_error(model.predict(trainset.as_matrix()), 
                                                          train_y.as_matrix())))
    ## and write it to a .csv file
    df = pd.DataFrame({'id': ids, 'loss': prediction})
    df.to_csv('./BaggingKerasSubmission.csv', index = False)

if __name__ == "__main__":

    train_path = './train.csv'
    test_path = './test.csv'
    n_estimators=10
    n_folds=5

    from keras.models import Sequential
    from keras.layers import Dense, Activation, BatchNormalization, Dropout
    from keras.layers.advanced_activations import PReLU
    from keras.wrappers.scikit_learn import KerasRegressor

    # let's try out with a very simple model
    def baseline_model(optimizer='adadelta', layer1=1024, layer2=512,
                       layer3=256,layer4=128,dropout_rate=0.5):
        # create model
        model = Sequential()
        
        model.add(Dense(512, input_dim=1190, 
                        init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        
        model.add(Dense(256, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate+.3))
        
        model.add(Dense(128, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate+.15))
        
        model.add(Dense(50, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, init='he_normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer=optimizer,
                      metrics=['mean_absolute_error'])
        return model

    layer1=[1024,1024,512,512,256,256,128,128,64,64]
    layer2=[512,256,256,128,128,64,64,32,32,16]
    layer3=[256,128,128,64,64,32,32,16,16,8]
    layer4=[128,64,64,32,32,16,16,8,8,4]
    dropout_rate = [0,.1]*5

    estimator_params = dict(layer1=layer1,
                            layer2=layer2,
                            layer3=layer3,
                            layer4=layer4,
                            dropout_rate=dropout_rate)

    # and the scoring function
    from sklearn.metrics import make_scorer, mean_absolute_error
    scoring = make_scorer(mean_absolute_error)
    main(train_path, test_path, n_estimators, estimator_params, scoring, n_folds)