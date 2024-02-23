import pandas as pd
import os
from matplotlib import pyplot as plt


def plot_final_metrics(data, save_path):
    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    f1_score_df = df.pivot(index="input", columns="hpo_strategy", values="f1_score")
    f1_score_df.plot(kind="line", ax=ax1, linestyle='dashed')
    ax1.set_ylabel("F1 Score")
    ax1.set_xlabel("Dataset")
    ax1.set_title("F1 Score Comparison: GRASP vs. Hyperband")
    ax1.legend(title="HPO Strategy results")

    time_df = df.pivot(index="input", columns="hpo_strategy", values="evaluation_time")
    time_df.plot(kind="line", ax=ax2)
    ax2.set_ylabel("Evaluation Time (s)")
    ax2.set_xlabel("Dataset")
    ax2.set_title("Evaluation Time Comparison: GRASP vs. Hyperband")
    ax2.legend(title="HPO Strategy results")

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path+"/final_metrics.png", format='png')
    # plt.savefig("src/"+save_path+"/final_metrics_test.png", format='png')
    df.to_csv(save_path+"/table.csv", index=False)
    # df.to_csv("src/"+save_path+"/table_test.csv", index=False)


def plot_evolution_through_time(data, dataset_names, output_path):
    for dataset_name in dataset_names:
        plt.figure(figsize=(10, 6))
        for item in data:
            if item['input'] == dataset_name:
                strategy, f1_scores, times = item["hpo_strategy"], item["evolution_through_time"][0], item["evolution_through_time"][1]
                plt.plot(times, f1_scores, label=f"{strategy}")

        plt.title(f'Evolution Through Time - {dataset_name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)

        #list all directories and files
        # print("Before saving current figure")
        # plt.savefig(f'src/{output_path}/{dataset_name}test.png')
        plt.savefig(f'{output_path}/{dataset_name}Evolution.png')
