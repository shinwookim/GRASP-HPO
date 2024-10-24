import pandas as pd
from pathlib import Path
import os

from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target


class ErenoProcessor:
    @staticmethod
    def load_data():
        src_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_folder = os.path.join(src_folder, "data")
        data_df = pd.read_csv(os.path.join(input_folder, "hybridGoose.csv"), sep=',')

        columns_to_drop = ["stDiff", "sqDiff", "gooseLengthDiff", "cbStatusDiff", "apduSizeDiff", "frameLengthDiff", "timestampDiff", "tDiff", "timeFromLastChange", "delay"]
        data = (data_df
                .drop(columns=columns_to_drop, errors='ignore')
                .drop(columns=['@class@']).apply(pd.to_numeric, errors='coerce')
                .dropna(axis=1))

        label_encoder = LabelEncoder()
        labels = data_df['@class@']
        target = label_encoder.fit_transform(labels)

        return Dataset(data, target)
