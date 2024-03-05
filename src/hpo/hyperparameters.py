HYPERPARAMETERS = {
    'default': {
        'max_depth': 6,
        'colsample_bytree': 1,
        'reg_lambda': 1,
        'subsample': 1,
        "min_child_weight": 1,
        "learning_rate": 0.3,
        "gamma": 0
    },
    'search_space': {
        'max_depth': (3, 10),
        'colsample_bytree': (0.5, 1),
        'reg_lambda': (0.01, 1.0),
        'subsample': (0.5, 1.0),
        "min_child_weight": (1, 10),
        "learning_rate": (1e-3, 0.3),
        "gamma": (0, 1)
    }
}


def get_hyperparameters(set_name='default'):
    return HYPERPARAMETERS.get(set_name, {})
