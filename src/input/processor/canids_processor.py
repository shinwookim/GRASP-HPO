import dask.dataframe as dd
import os
from pathlib import Path
from src.input.processor.dsobj import Dataset
from dask_ml.preprocessing import LabelEncoder

class canidsProcessor:
    @staticmethod
    def load_data():
        df1 = dd.read_parquet(os.path.join(Path(__file__).parent.parent.absolute(), "data", "dump1.parquet"), engine = 'pyarrow')
        df2 = dd.read_parquet(os.path.join(Path(__file__).parent.parent.absolute(), "data", "dump6-repl-360-479.99999.parquet"), engine = 'pyarrow')
        df = dd.concat([df1, df2], ignore_index=True)

        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(df1['112_CUR_GR'])

        return Dataset(df, target)