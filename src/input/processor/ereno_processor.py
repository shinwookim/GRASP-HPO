import pandas as pd
from pathlib import Path

from sklearn.preprocessing import LabelEncoder


class Dataset:
    def __init__(self, data, target) -> None:
        self.data = data
        self.target = target


class ErenoProcessor:
    @staticmethod
    def load_data():
        data_df = pd.read_csv("./input/data/hybridGoose.csv", sep=',')

        data = data_df.drop(columns=['@class@']).apply(pd.to_numeric, errors='coerce').dropna(axis=1)

        label_encoder = LabelEncoder()
        labels = data_df['@class@']
        target = label_encoder.fit_transform(labels)

        return Dataset(data, target)
