import pandas as pd

from sklearn.preprocessing import LabelEncoder

from src.input.processor.dataset import Dataset


class ErenoProcessor:
    @staticmethod
    def load_data():
        data_df = pd.read_csv("./input/data/hybridGoose.csv", sep=',')

        columns_to_drop = ["stDiff", "sqDiff", "gooseLengthDiff", "cbStatusDiff", "apduSizeDiff", "frameLengthDiff", "timestampDiff", "tDiff", "timeFromLastChange", "delay"]
        data = (data_df
                .drop(columns=columns_to_drop, errors='ignore')
                .drop(columns=['@class@']).apply(pd.to_numeric, errors='coerce')
                .dropna(axis=1))

        label_encoder = LabelEncoder()
        labels = data_df['@class@']
        target = label_encoder.fit_transform(labels)

        return Dataset(data, target)
