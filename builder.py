import data
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from nyoka import skl_to_pmml
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np

df = data.build_dataset(10000)


def build_LR_pipeline(inputs, outputs):
    pipeline = Pipeline([
        ("regressor", LinearRegression())
    ])
    pipeline.fit(inputs, outputs)
    return pipeline


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))


def RF_estimation(inputs, outputs,
                  estimator_steps=10,
                  depth_steps=10,
                  min_samples_split=None,
                  min_samples_leaf=None):
    # hyper-parameter estimation
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=estimator_steps)]
    max_depth = [int(x) for x in np.linspace(10, 100, num=depth_steps)]
    max_depth.append(None)
    if not min_samples_split:
        min_samples_split = [2, 5, 10]
    if not min_samples_leaf:
        min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(), param_distributions=random_grid,
                                   n_iter=100, scoring='neg_mean_absolute_error',
                                   cv=3, verbose=1, random_state=42, n_jobs=-1)
    rf_random.fit(inputs, outputs)
    best_random = rf_random.best_estimator_
    print(best_random)
    return best_random


def build_RF_pipeline(inputs, outputs, rf=None):
    if not rf:
        rf = RandomForestRegressor()
    pipeline = Pipeline([
        ("regressor", rf)
    ])
    pipeline.fit(inputs, outputs)
    return pipeline


def build_ANN_model(n_inputs):
    # create a sequential model
    model = Sequential()
    model.add(Dense(n_inputs, input_dim=n_inputs, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


if __name__ == '__main__':
    # Dispute Risk (DR) model

    # dispute risk data
    DR_outputs = df['dispute_risk']
    DR_inputs = df[['amount', 'holder_index']]

    # split dataset
    DR_X_train, DR_X_test, DR_y_train, DR_y_test = train_test_split(DR_inputs, DR_outputs, test_size=0.4,
                                                                    random_state=23)

    # save the testing datasets
    DR_X_test.to_csv("data/dispute_risk_test.csv", columns=['amount', 'holder_index'], index=False)
    df.to_csv("data/dispute_risk.csv", columns=['amount', 'holder_index', 'dispute_risk'], index=False)

    # dispute risk linear regression
    DR_linear_regression_pipeline = build_LR_pipeline(DR_X_train, DR_y_train)

    # save PMML model
    skl_to_pmml(DR_linear_regression_pipeline, ['amount', 'holder_index'], 'dispute_risk',
                "models/dispute_risk_linear_regression.pmml")

    # dispute risk random forest
    DR_rf = RF_estimation(DR_X_train, DR_y_train, estimator_steps=5, depth_steps=5)
    DR_random_forest_pipeline = build_RF_pipeline(DR_X_train, DR_y_train, DR_rf)

    # save PMML model
    skl_to_pmml(DR_random_forest_pipeline, ['amount', 'holder_index'], 'dispute_risk',
                "models/dispute_risk_random_forest.pmml")

    # dispute risk ANN
    DR_ANN_model = build_ANN_model(2)
    DR_ANN_model.fit(DR_X_train, DR_y_train, nb_epoch=10, batch_size=16, validation_split=0.1, verbose=2)
    preds = DR_ANN_model.predict_classes(DR_X_test, verbose=0)
    # Card Holder (CH) model

    # data
    CH_outputs = df['holder_risk']
    CH_inputs = df[['age', 'holder_index', 'incidents']]

    # split dataset
    CH_X_train, CH_X_test, CH_y_train, CH_y_test = train_test_split(CH_inputs, CH_outputs, test_size=0.4,
                                                                    random_state=23)

    # save the testing dataset
    CH_X_test.to_csv("data/card_holder_risk_test.csv", columns=['age', 'holder_index', 'incidents'], index=False)
    df.to_csv("data/holder_risk.csv", columns=['age', 'holder_index', 'incidents', 'holder_risk'], index=False)

    # dispute risk linear regression
    CH_linear_regression_pipeline = build_LR_pipeline(CH_X_train, CH_y_train)

    # save PMML model
    skl_to_pmml(CH_linear_regression_pipeline, ['age', 'holder_index', 'incidents'], 'holder_risk',
                "models/card_holder_risk_linear_regression.pmml")

    # dispute risk random forest
    CH_rf = RF_estimation(CH_X_train, CH_y_train, estimator_steps=5, depth_steps=5)
    CH_random_forest_pipeline = build_RF_pipeline(CH_X_train, CH_y_train)

    # save PMML model
    skl_to_pmml(CH_random_forest_pipeline, ['age', 'holder_index', 'incidents'], 'holder_risk',
                "models/card_holder_risk_random_forest.pmml")
