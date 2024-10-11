HYPERPARAMETERS = {
    'default': {
        'penalty': 'l2',
        'loss': 'squared_hinge',
        'dual': 'auto',
        'tol': 1e-4,
        'C': 1.0,
        'multi_class': 'ovr',
        'fit_intercept': True,
        'intercept_scaling': 1,
        'class_weight': None,
        'verbose': 0,
        'random_state': None,
        'max_iter': 1000
    },
    'search_space': {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'tol': (1e-5, 1e-3),
        'C': (1e-3, 1e3),
        'multi_class': ['ovr', 'crammer_singer'],
        'fit_intercept': [True, False],
        'intercept_scaling': (1, 10),
        'class_weight': [None, 'balanced'],
        'verbose': [0, 1],
        'random_state': [None, 42],
        'max_iter': (1000, 5000)
    }
}


def get_hyperparameters(set_name='default'):
    return HYPERPARAMETERS.get(set_name, {})
