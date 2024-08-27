HYPERPARAMETERS = {
    'default': {
        'n_estimators': 100,
        'max_depth': 6,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt',
        'bootstrap': True,
    },
    'search_space': {
        'n_estimators': (10, 200),
        'max_depth': (3, 30),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
    }
}


def get_hyperparameters(set_name='default'):
    return HYPERPARAMETERS.get(set_name, {})