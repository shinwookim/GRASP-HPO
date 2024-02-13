import pandas as pd
import matplotlib.pyplot as plt


def plot_final_metrics(data, save_path):
    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    f1_score_df = df.pivot(index="input", columns="hpo_strategy", values="f1_score")
    f1_score_df.plot(kind="line", ax=ax1, marker='o', linestyle='dashed')
    ax1.set_ylabel("F1 Score")
    ax1.set_xlabel("Dataset")
    ax1.set_title("F1 Score Comparison: GRASP vs. Hyperband")
    ax1.legend(title="HPO Strategy results")

    time_df = df.pivot(index="input", columns="hpo_strategy", values="evaluation_time")
    time_df.plot(kind="line", ax=ax2, marker='o')
    ax2.set_ylabel("Evaluation Time (s)")
    ax2.set_xlabel("Dataset")
    ax2.set_title("Evaluation Time Comparison: GRASP vs. Hyperband")
    ax2.legend(title="HPO Strategy results")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path + "final_metrics.png", format='png')


if __name__ == '__main__':
    # Load the data from the CSVs
    data_1 = pd.read_csv("breast.csv")
    data_2 = pd.read_csv("Digits.csv")
    data_3 = pd.read_csv("ereno.csv")
    data_4 = pd.read_csv("iris.csv")
    data_5 = pd.read_csv("wine.csv")

    combined_data = pd.concat([data_1, data_2, data_3, data_4, data_5])

    # Plot and save the metrics
    plot_final_metrics(combined_data, "")
