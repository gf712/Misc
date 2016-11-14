#!/usr/bin/env python

import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def main(train_path, test_path, n_estimators, model, scoring, n_folds):

    ## load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    print('Loaded data')

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
    print('Size of test set: {}'.format(len(testset)))
    print('Size of train set: {}'.format(len(trainset)))

    ## instantiate variables
    prediction = np.zeros((len(testset)))

    ## do the bagging kfold
    folds = KFold(n_splits = n_folds, shuffle = True)
    validation_score = np.zeros((n_folds))

    for i, (train_index, test_index) in tqdm(enumerate(folds.split(trainset))):

        train_X_nfolds = trainset.ix[train_index,:]
        train_y_nfolds = train_y[train_index]

        ## get number of data points per fold
        estimator_size  = int(len(train_X_nfolds) / n_estimators)

        ## keep track of score per fold in validation set
        validation_X = trainset.ix[test_index,:]
        validation_y = train_y[test_index]
        validation_score_i = 0
        
        ## start the bagging 
        for x in range(n_estimators):

            ## draw data randomly with replacement
            index = np.random.choice(len(train_X_nfolds), size=estimator_size, replace=True)
            train_Xi = train_X_nfolds.iloc[index,:]
            train_yi = train_y_nfolds.iloc[index]

            ## train model
            model.fit(train_Xi.as_matrix(), train_yi.as_matrix())

            ## update test prediction
            prediction += model.predict(testset.as_matrix())
            validation_score_i += mean_absolute_error(model.predict(validation_X.as_matrix()),
                                                      validation_y.as_matrix())

        ## store the score of validation set
        validation_score[i] = validation_score_i / n_estimators

    ## get final prediction
    prediction /= (n_folds*n_estimators)
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
    n_folds=3

    from keras.models import Sequential
    from keras.layers import Dense, Activation, BatchNormalization, Dropout
    from keras.layers.advanced_activations import PReLU
    from keras.wrappers.scikit_learn import KerasRegressor

    # let's try out with a very simple model
    def baseline_model(optimizer='adadelta',dropout_rate=0.5):
        # create model
        model = Sequential()
        
        model.add(Dense(1024, input_dim=1190, 
                        init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        
        model.add(Dense(512, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate+.2))
        
        model.add(Dense(256, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate+.1))
        
        model.add(Dense(128, init='he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        model.add(Dense(1, init='he_normal'))
        # Compile model
        model.compile(loss='mean_absolute_error', optimizer=optimizer)
        return model

    model = KerasRegressor(build_fn=baseline_model, nb_epoch=20, 
                           batch_size=500, verbose=1,
                           dropout_rate=.1)

    # and the scoring function
    from sklearn.metrics import make_scorer, mean_absolute_error
    scoring = make_scorer(mean_absolute_error)

    main(train_path, test_path, n_estimators, model, scoring, n_folds)