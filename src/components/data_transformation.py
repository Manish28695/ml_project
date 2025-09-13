import sys 
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import Custom_exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''This function is responsible for data transformation'''

        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            numerical_pipeline = Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),
                       ('scaler',StandardScaler())
                        ]

            )
            categorical_pipeline = Pipeline(
                steps = [('imputer',SimpleImputer(strategy='most_frequent')),
                         ('one_hot_encoder',OneHotEncoder()),
                         ('scaler',StandardScaler(with_mean=False))
                         ]
            )
            logging.info('numerical columns standard scaling completed')
            logging.info('categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                [('numerical_pipeline',numerical_pipeline,numerical_columns),
                ('categorical_pipeline',categorical_pipeline,categorical_columns)]
            )
            return preprocessor
        except Exception as e:
            raise Custom_exception(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read the train and test data completed')

            logging.info('obtaining the preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'

            numerical_columns = ['writing_score','reading_score']

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('applying the preprocessing object on training and testing dataframes')

            input_feature_train_df = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_df,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_df,np.array(target_feature_test_df)]

            logging.info('saved the preprocessing object')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise Custom_exception(e,sys)
        

        


