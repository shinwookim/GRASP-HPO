import sklearn.datasets
import sklearn.metrics
import os
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
import xgboost as xgb

from ray.tune.integration.xgboost import TuneReportCheckpointCallback
import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

def train_breast_cancer(config: dict):
    # This is a simple training function to be passed into Tune
    # Load dataset
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    # Split into train and test set
    train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2)
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
        feval=f1_eval,
    )

def f1_eval(preds, dtrain):
    labels = dtrain.get_label()
    return "f1_score", sklearn.metrics.f1_score(labels, preds > 0.5)

def get_best_model_checkpoint(results):
    best_bst = xgb.Booster()
    best_result = results.get_best_result()

    with best_result.checkpoint.as_directory() as best_checkpoint_dir:
        best_bst.load_model(os.path.join(best_checkpoint_dir, "model.xgb"))
    # accuracy = 1.0 - best_result.metrics["eval-error"]
    print(f"Best model parameters: {best_result.config}")
    # best f1_score
    print(f"Best model f1_score: {best_result.metrics['eval-f1_score']:.4f}")
    # print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst


def tune_xgboost(smoke_test=False):
    search_space = {
        # You can mix constants with search space objects.
        "objective": "binary:logistic",
        "disable_default_eval_metric": 1,
        "max_depth": hp.randint("max_depth", 3, 10),
        "min_child_weight": hp.uniform("min_child_weight", 1, 10),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "learning_rate": hp.loguniform("learning_rate", 1e-3, 1.0),
        "gamma": hp.uniform("gamma", 0, 1),
        "n_estimators": hp.choice("n_estimators", [50, 500]),
        "num_boost_round": hp.choice("num_boost_round", [10, 50, 100]),
    }
    # This will enable aggressive early stopping of bad trials.
    scheduler = ASHAScheduler(
        max_t=10, grace_period=1, reduction_factor=2  # 10 training iterations
    )

    algo = HyperOptSearch(space=search_space, metric="eval-f1_score", mode="max")
    # algo = ConcurrencyLimiter(algo, max_concurrent=4)

    tuner = tune.Tuner(
        train_breast_cancer,
        tune_config=tune.TuneConfig(
            metric="eval-f1_score",
            mode="max",
            scheduler=scheduler,
            search_alg=algo,
            num_samples=10,
        ),

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