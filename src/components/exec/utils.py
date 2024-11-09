import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from exec.exception import CustomException

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    
    except Exception as e:
        raise CustomException(e,sys)
        # pass

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            test_model_score = r2_score(y_test, y_pred)
            results[name] = test_model_score
        
        return results
        
    except Exception as e:
        raise CustomException(e, sys)
        # pass