import joblib
import os
import lightgbm as lgb
from catboost import CatBoostClassifier
from config import LGBM_PARAMS, CATBOOST_PARAMS, MODEL_DIR


def train_lightgbm(X_train, y_train):
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "lgbm_model.pkl"))
    return model


def train_catboost(X_train, y_train):
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train)
    model.save_model(os.path.join(MODEL_DIR, "catboost_model.cbm"))
    return model
