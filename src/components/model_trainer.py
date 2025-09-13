import os 
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier

from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Custom_exception
from src.logger import logging
from src.utils import save_object,evaluate_models  


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train,X_test, y_test = (train_array[:,:-1], train_array[:,-1],
                                              test_array[:,:-1], test_array[:,-1])


            logging.info('Loading preprocessor object')
            
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "Logistic Regression": LogisticRegression(),
                "XGB Classifier": XGBRegressor(),
                "CatBoost Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier()
            }

            model_report:dict = evaluate_models(X = X_train, y = y_train, models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Custom_exception("No best model found")
            logging.info(f'Best model found on both training and testing dataset')


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise Custom_exception(e, sys)
