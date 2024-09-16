import torch

def load_model(cfg):
    if cfg.Model.model_type == 'experiment':
        model = load_experimental_model(cfg)
    elif cfg.Model.model_type == 'public':
        model = load_public_model(cfg)
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    return model

# TODO: Delete or Convert to Stable
def load_experimental_model(cfg):

    raise NotImplementedError





def load_public_model(cfg):
    if cfg.Model.model_name == 'SCL':
        from models.public_models.SCL import SCL
        model = SCL(n_classes=cfg.Data.n_classes, **cfg.Model)
        print('success to init SCLv2')


    else:
        raise NotImplementedError

    return model

