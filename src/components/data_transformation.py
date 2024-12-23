import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    
    def get_transformer(self):
        try:
            numerical_col=['reading score', 'writing score']
            categorical_cols=['gender',
                                'race/ethnicity',
                                'parental level of education',
                                'lunch',
                                'test preparation course'
                            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(
               steps= [
                   ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]
            )
            logger.info("numerical columns pipeline created")
            logger.info("categorical columns pipeline created")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_col),
                    ("categorical_pipelines",cat_pipeline,categorical_cols)

                ]
            )

            return preprocessor
            

    
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            print("hello")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logger.info("read train and test data")

            preprocessor_obj=self.get_transformer()

            target_column='math score'
            numerical_col=['reading score', 'writing score']
            input_features_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]


            input_features_test_df=test_df.drop(columns=[target_column],axis=1)
            target_features_test_df=test_df[target_column]


            logger.info("Applying preprocessing object")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_features_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_features_test_df)]


            logger.info("saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj


            )

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path

            )
                                                  

        except Exception as e:
            raise CustomException(e,sys)
        