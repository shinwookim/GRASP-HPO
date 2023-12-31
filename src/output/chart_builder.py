import pandas as pd
from matplotlib import pyplot as plt


def plot(data, save_path):
    df = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    f1_score_df = df.pivot(index="input", columns="hpo_strategy", values="f1_score")
    f1_score_df.plot(kind="line", ax=ax1, marker='o')
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

    plt.savefig(save_path+"/chart.png", format='png')
    df.to_csv(save_path+"/table.csv", index=False)
