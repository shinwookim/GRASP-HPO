import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.hpo.hpo_strategy import HPOStrategy


class Default(HPOStrategy):
    def hyperparameter_optimization(self, x_train, y_train, x_val, y_val):
        # Setup default parameters, these can be customized as needed
        params = {
            'n_estimators': 100,  # Number of trees in the forest
            'max_depth': None,  # The maximum depth of the tree
            'min_samples_split': 2,  # Minimum number of samples required to split an internal node
            'min_samples_leaf': 1,  # Minimum number of samples required to be at a leaf node
            'max_features': 'sqrt'  # Number of features to consider when looking for the best split
        }

        # Initialize the RandomForest model
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            random_state=42  # Ensuring a deterministic behaviour
        )

        # Train the RandomForest model
        model.fit(x_train, y_train)

        # Evaluate the model using the validation set
        y_pred = model.predict(x_val)
        f1 = f1_score(y_val, y_pred, average="weighted")  # Compute F1 score on validation set

        # No need for cumulative time and other metrics in this default scenario
        return model, f1, None

