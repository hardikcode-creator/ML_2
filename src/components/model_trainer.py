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
                    
            params = {
                "Linear Regression": {},  # Linear regression generally has no hyperparameters to tune.

                "K Neighbours": {
                    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider.
                    'weights': ['uniform', 'distance'],  # Weighting strategy for neighbors.
                    'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics.
                },

                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],  # Strategy for choosing split.
                    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree.
                    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node.
                    'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be a leaf node.
                },

                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of trees in the forest.
                    'criterion': ['squared_error', 'absolute_error'],  # Splitting criteria.
                    # 'max_depth': [None, 10, 20, 30],  # Maximum depth of trees.
                    # 'min_samples_split': [2, 5, 10],  # Minimum samples for splitting.
                    # 'min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf.
                    # 'bootstrap': [True, False]  # Whether bootstrap samples are used.
                },

                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],  # Step size shrinkage.
                    # 'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting stages.
                    # 'max_depth': [3, 5, 7],  # Maximum depth of trees.
                    # 'subsample': [0.6, 0.8, 0.9],  # Fraction of samples used for training.
                    'colsample_bytree': [0.6, 0.8, 1.0],  # Fraction of features used per tree.
                    'gamma': [0, 0.1, 0.2],  # Minimum loss reduction required to make a split.
                },

                "CatBooosting Regressor": {
                    'iterations': [100, 200, 300],  # Number of boosting iterations.
                    'learning_rate': [0.1, 0.01, 0.05],  # Step size shrinkage.
                    'depth': [3, 5, 7, 9],  # Depth of the tree.
                    'l2_leaf_reg': [1, 3, 5, 7],  # L2 regularization term.
                },

                "AdaBoostRegressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting stages.
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],  # Step size shrinkage.
                    'loss': ['linear', 'square', 'exponential'],  # Loss function for boosting.
                },
                
                "GradientBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],  # Step size shrinkage.
                    # 'subsample': [0.6, 0.8, 0.75, 0.85, 0.9],  # Fraction of samples used for training.
                    # 'n_estimators': [8, 16, 32, 64, 128, 256],  # Number of boosting stages.
                    # 'max_depth': [3, 5, 7],  # Maximum depth of individual estimators.
                    'min_samples_split': [2, 5],  # Minimum samples to split a node.
                    'min_samples_leaf': [1, 2],  # Minimum samples in a leaf.
                },

                }

            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,params=params)
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




