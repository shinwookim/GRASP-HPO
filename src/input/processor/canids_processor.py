import pandas as pd
from pathlib import Path
import os

from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target


class CanidsProcessor:
    @staticmethod
    def load_data():
        src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_folder = os.path.join(src_folder, "data")
        data_df_1 = pd.read_parquet(os.path.join(input_folder, "dump1.parquet"))
        data_df_2 = pd.read_parquet(os.path.join(input_folder, "dump6-repl-360-479.99999.parquet"))
        data_df = pd.concat([data_df_1, data_df_2], ignore_index=True)

        #remove data_df_1 and data_df_2 from memory
        del data_df_1
        del data_df_2

        encoder = LabelEncoder()
        data_df['112_CUR_GR'] = encoder.fit_transform(data_df['112_CUR_GR'])
        data = data_df.drop('label', axis=1)
        target = data_df['label']

        return Dataset(data, target)
