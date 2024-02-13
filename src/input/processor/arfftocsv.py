import pandas as pd
from scipy.io import arff


def merge_arff_to_csv(arff_file1, arff_file2, csv_file_name):
    try:
        # Load ARFF files
        data1, meta1 = arff.loadarff(arff_file1)
        data2, meta2 = arff.loadarff(arff_file2)

        # Convert ARFF data to Pandas DataFrames
        df1 = pd.DataFrame(data1)
        df2 = pd.DataFrame(data2)

        # Merge DataFrames
        merged_df = pd.merge(df1, df2, on='common_column_name', how='inner')

        # Handle non-numeric values
        merged_df = merged_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        # Drop columns with NaN values
        merged_df = merged_df.dropna(axis=1)

        # Save DataFrame to CSV file in the same directory as the ARFF files
        csv_file_path = f"./{csv_file_name}.csv"
        merged_df.to_csv(csv_file_path, index=False)

        print(f"CSV file saved at: {csv_file_path}")

    except ValueError as e:
        print(f"Error: {e}")
        print(f"Non-numeric values found in ARFF files. Check the data in the specified columns.")


if __name__ == "__main__":
    arff_file1 = '../data/hibrid_dataset_GOOSE_test.arff'
    arff_file2 = '../data/hibrid_dataset_GOOSE_train.arff'
    csv_file_name = '../data/ereno'

    merge_arff_to_csv(arff_file1, arff_file2, csv_file_name)
