import pandas as pd
from pathlib import Path
from dsobj import Dataset

from sklearn.preprocessing import LabelEncoder

class canidsProcessor:
    @staticmethod
    def load_data():
        df1 = pd.read_parquet("./input/data/dump1.parquet")
        df2 = pd.read_parquet("./input/data/dump6-repl-360-479.99999.parquet")

        df = pd.concat([df1, df2])

        label_encoder = LabelEncoder()
        labels = df['112_CUR_GR']
        target = label_encoder.fit_transform(labels)

        return Dataset(df, target)