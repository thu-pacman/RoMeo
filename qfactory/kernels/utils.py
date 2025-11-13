import itertools

def span_tuning_space(tunable_keys):
    combinations = itertools.product(*tunable_keys.values())
    keys = tunable_keys.keys()
    result = tuple(dict(zip(keys, combo)) for combo in combinations)
    return result


if __name__ == '__main__':
    # test span_tuning_space
    tunable_keys = {
        'A': [1, 2, 3],
        'B': [128, 256],
    }
    print(span_tuning_space(tunable_keys))