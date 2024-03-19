import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.input.processor.dataset import Dataset


class PowerSystemAttackProcessor:
    @staticmethod
    def load_data(filepath="./input/data/PowerSystemAttackDatasets.csv"):
        data_df = pd.read_csv(filepath)

        columns_to_drop = ['control_panel_log1', 'control_panel_log2', 'control_panel_log3', 'control_panel_log4',
                           'relay1_log', 'relay2_log', 'relay3_log', 'relay4_log',
                           'snort_log1', 'snort_log2', 'snort_log3', 'snort_log4']
        data_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

        # Replace inf/-inf with NaN because xgboost is not able to handle inf values
        data_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Impute missing values (numeric columns with mean, might not be the best approach)
        numeric_columns = data_df.select_dtypes(include=['float64', 'int']).columns
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        data_df[numeric_columns] = imputer.fit_transform(data_df[numeric_columns])

        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(data_df['marker'])
        data_df.drop(columns=['marker'], inplace=True)

        scaler = StandardScaler()
        scaled_features = pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)

        return Dataset(scaled_features, target)
