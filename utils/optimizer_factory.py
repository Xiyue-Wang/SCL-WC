import torch

def create_optimizer(model, cfg):
    if cfg.Train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.Train.lr,
                                     weight_decay=cfg.Train.reg)
    else:
        raise NotImplementedError

    return optimizer