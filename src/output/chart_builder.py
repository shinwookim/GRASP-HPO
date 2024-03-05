import pandas as pd
import os
from matplotlib import pyplot as plt


def plot_final_metrics(data, save_path):
    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(14, 6))

    f1_score_df = df.pivot(index="input", columns="hpo_strategy", values="f1_score")

    f1_score_df.plot(kind="line", ax=ax1, marker='o', linestyle='dashed')

    ax1.set_ylabel("F1 Score")
    ax1.set_xlabel("Dataset")
    ax1.set_title("F1 Score Comparison")

    ax1.legend(title="HPO Strategy")

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{save_path}/final_metrics.png", format='png')
    df.to_csv(f"{save_path}/table.csv", index=False)


def plot_evolution_through_time(data, dataset_names, output_path):
    for dataset_name in dataset_names:
        plt.figure(figsize=(10, 6))
        for item in data:
            if item['input'] == dataset_name:
                strategy = item["hpo_strategy"]
                plt.plot(item["cumulative_time"], item["f1_scores"], marker='o', label=f"{strategy}")

        plt.title(f'Evolution Through Time - {dataset_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        #list all directories and files
        # print("Before saving current figure")
        # plt.savefig(f'src/{output_path}/{dataset_name}test.png')
        plt.savefig(f'{output_path}/{dataset_name}Evolution.png')
