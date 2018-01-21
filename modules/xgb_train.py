import numpy as np
import pandas as pd
import xgboost as xgb


def feature_importance(xgb_model, feature_names, threshold=.0):
    #  Returns lists with low-correlation and high-correlation features in XGBoost xgb_model
    #  If threshold is int, the first n features are important, the rest unimportant

    importances = np.array([feature_names, xgb_model.feature_importances_]).T
    importances = pd.DataFrame(importances, columns=['feature', 'importance'])
    importances.sort_values('importance', ascending=False, inplace=True)
    ranked = list(importances['feature'])

    if type(threshold) == int:
        threshold = min(threshold, importances.shape[0])
        unimportant = list(importances['feature'][threshold:])
    else:
        unimportant = list(importances[importances['importance'] <= threshold]['feature'])

    return importances, ranked, unimportant


def xgb_train(X_train, X_eval, X_test, y_eval, y_train, params):

    #  Initiate model
    model = xgb.XGBClassifier(**params)

    #  Training model using training data (takes a while!)
    X_train, y_train, feature_names = X_train.as_matrix(), y_train.as_matrix().ravel(), X_train.columns
    eval_set = [(X_eval.as_matrix(), y_eval.as_matrix().ravel())]
    model.fit(X_train, y_train, eval_set=eval_set, eval_metric=params['eval_metric'],
              early_stopping_rounds=params['early_stopping_rounds'])

    #  Read important and unimportant features
    features = {}
    features['names'] = feature_names
    features['importances'], features['ranked'] = feature_importance(model, feature_names, 10)[:2]
    features['unimportant'] = feature_importance(model, feature_names, .0)[2]

    return model, features
