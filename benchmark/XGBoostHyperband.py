import xgboost as xgb
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import sklearn.datasets
import sklearn.metrics
from ray.tune.integration.xgboost import TuneReportCheckpointCallback


# Define your XGBoost training function
def train_xgboost(config):
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    dtrain = xgb.DMatrix(data, label=labels)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": config["max_depth"],
        "min_child_weight": config["min_child_weight"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "eta": config["learning_rate"],
        "gamma": config["gamma"],
        "n_estimators": config["n_estimators"],
        "seed": 42,
    }

    # Train XGBoost model
    evals = [(dtrain, "train")]
    bst = xgb.train(
        params,
        dtrain,
        config["num_boost_round"],
        evals=evals,
        verbose_eval=False,
        callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
    )

    # Predict
    preds = bst.predict(dtrain)
    threshold = 0.5  # You can adjust this threshold as needed
    binary_preds = [1 if p > threshold else 0 for p in preds]

    # Calculate F1 score
    f1 = sklearn.metrics.f1_score(labels, binary_preds)

    return {"f1_score": f1}


if __name__ == "__main__":
    ray.init(local_mode=False)  # Initialize Ray (or specify your cluster configuration)

    # Define the hyperparameter search space
    config = {
        "max_depth": tune.randint(3, 10),
        "min_child_weight": tune.uniform(1, 10),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "learning_rate": tune.loguniform(1e-3, 0.1),
        "gamma": tune.uniform(0, 1),
        "n_estimators": tune.choice([100, 200, 300]),
        "num_boost_round": tune.choice([10, 50, 100]),
    }

    # Define the ASHA scheduler for hyperparameter optimization
    scheduler = ASHAScheduler(
        max_t=100,  # Maximum number of training iterations
        grace_period=10,  # Minimum number of iterations for each trial
        reduction_factor=2,  # Factor by which trials are pruned
    )

    # Run the hyperparameter search
    analysis = tune.run(
        train_xgboost,
        config=config,
        num_samples=10,  # Number of trials to run
        metric="f1_score",  # Metric to optimize
        mode="max",  # Maximize the F1 score
        scheduler=scheduler,
    )

    # Get the best configuration
    best_trial = analysis.get_best_trial(metric="f1_score", mode="max")
    best_f1 = best_trial.last_result["f1_score"]
    print("Best hyperparameters:", best_trial.config)
    print(f"Best model total accuracy: {best_f1}")
    # Close Ray
    ray.shutdown()
