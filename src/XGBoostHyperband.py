# Adapted from https://docs.ray.io/en/latest/tune/examples/tune-xgboost.html
import sklearn.datasets
import sklearn.metrics
import os
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ray import train, tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback


def train_breast_cancer(config: dict):
    # This is a simple training function to be passed into Tune
    # Load dataset
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.25)
    # Build input matrices for XGBoost
    train_set = xgb.DMatrix(train_x, label=train_y)
    test_set = xgb.DMatrix(test_x, label=test_y)
    # Train the classifier, using the Tune callback
    xgb.train(
        config,
        train_set,
        evals=[(test_set, "eval")],
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
    )


def get_best_model_checkpoint(results):
    best_bst = xgb.Booster()
    best_result = results.get_best_result()

    with best_result.checkpoint.as_directory() as best_checkpoint_dir:
        best_bst.load_model(os.path.join(best_checkpoint_dir, "model.xgb"))
    accuracy = 1.0 - best_result.metrics["eval-error"]
    print(f"Best model parameters: {best_result.config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst


def tune_xgboost(smoke_test=False):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "eta": tune.loguniform(1e-4, 1e-1),
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )

    tuner = tune.Tuner(
        train_breast_cancer,
        tune_config=tune.TuneConfig(
            metric="eval-logloss",
            mode="min",
            scheduler=scheduler,
            num_samples=1 if smoke_test else 10,
        ),
        param_space=search_space,
    )
    results = tuner.fit()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing"
    )
    args, _ = parser.parse_known_args()

    results = tune_xgboost(smoke_test=args.smoke_test)

    # Load the best model checkpoint.
    best_bst = get_best_model_checkpoint(results)

    # You could now do further predictions with
    # best_bst.predict(...)
