def create_training_loop(cfg):
    if cfg.Train.train_function == 'classification_general':
        from training_methods.classification_general import train_loop

    elif cfg.Train.train_function == 'SCL':
        from training_methods.SCL import train_loop
    else:
        raise NotImplementedError

    return train_loop

def create_validation(cfg):
    if cfg.Train.val_function == 'classification_general':
        from training_methods.classification_general import validation
    else:
        raise NotImplementedError

    return validation


def create_evaluation(cfg):
    if cfg.Train.val_function == 'classification_general':
        from training_methods.classification_general import evaluation
    else:
        raise NotImplementedError

    return evaluation