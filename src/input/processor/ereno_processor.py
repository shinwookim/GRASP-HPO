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
        root_dir = Path(__file__).resolve().parent.parent.parent.parent
        data_df = pd.read_csv(root_dir / "data" / "erenoFull.csv", sep=',')

        columns_to_remove = ['ethDst', 'ethSrc', 'ethType', 'gooseAppid', 'TPID', 'gocbRef', 'datSet', 'goID', 'test', 'ndsCom', 'protocol']

        data_df = data_df.dropna(axis=1).drop(columns=columns_to_remove, errors='ignore')

        data = data_df.drop(columns=['@class@'])

        label_encoder = LabelEncoder()
        labels = data_df['@class@']
        target = label_encoder.fit_transform(labels)

        return Dataset(data, target)
