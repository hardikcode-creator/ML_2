import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={
             "Linear Regression":LinearRegression(),
              "K Neighbours":KNeighborsRegressor(),
              "Decision Tree":DecisionTreeRegressor(),
              "Random Forest Regressor":RandomForestRegressor(),
              "XGBRegressor":XGBRegressor(),
                    "CatBooosting Regressor":CatBoostRegressor(),
                 "AdaBoostRegressor":AdaBoostRegressor(),
                 "GradientBoost":GradientBoostingRegressor()

                }
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models)
            best_model_score=max(sorted(model_report.values()))

            #This finds the index of best model and then access the name of that model
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]


            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best Model",sys)
            logging.info(f"Best found model on both training and testing ")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                            obj=best_model
                        )
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            return r2


        except Exception as e:
            raise CustomException(e,sys)




