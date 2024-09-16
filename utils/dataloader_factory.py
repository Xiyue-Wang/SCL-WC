import os
import random
import shutil
import math

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

def stratified_sampling(cfg, result_dir):
    cv_root = os.path.join(result_dir, 'cross_validation_splits')
    os.makedirs(cv_root, exist_ok=True)
    df = pd.read_csv(cfg.Data.dataset['train_set'].csv_path)

    labels = set(list(df['label'].values))
    data = []
    for label in labels:
        class_data = list(df.values[df['label'].values == label])
        random.shuffle(class_data)
        data.append(class_data)

    fold_num = cfg.General.fold_num
    for i in range(fold_num):
        fold_data = []
        for class_data in data:
            fold_size = int(math.ceil(len(class_data)/fold_num + 0.5))
            fold_data += class_data[fold_size * i: fold_size * i + fold_size]
        pd.DataFrame(data=np.array(fold_data), columns=df.columns).to_csv(os.path.join(cv_root, f'split_{i}.csv'), index=False)





def create_dataloader(index, dataset, cfg, result_dir):
    if cfg.Data.dataset_name in ['BagDataset', 'TwoStreamBagDataset']:
        return create_bag_dataloader(index, dataset, cfg, result_dir)
    else:
        raise NotImplementedError

def create_bag_dataloader(index, dataset_name, cfg, result_dir):
    if dataset_name in ['train_set', 'val_set']:
        if cfg.General.exp_type == 'repeat':
            df = pd.read_csv(cfg.Data.dataset[dataset_name].csv_path)
        else:
            if isinstance(cfg.Data.split_dir, str) and not os.path.exists(os.path.join(result_dir, 'cross_validation_splits')):
                shutil.copytree(cfg.Data.split_dir, os.path.join(result_dir, 'cross_validation_splits'))

            if not os.path.exists(os.path.join(result_dir, 'cross_validation_splits')):
                stratified_sampling(cfg, result_dir)
            if dataset_name == 'train_set':
                df = []
                for i in range(cfg.General.fold_num):
                    if i != index:
                        df.append(pd.read_csv(os.path.join(result_dir, 'cross_validation_splits', f'split_{i}.csv')))
                df = pd.concat(df, axis=0)
            else:
                df = pd.read_csv(os.path.join(result_dir, 'cross_validation_splits', f'split_{index}.csv'))
    else:
        df = pd.read_csv(cfg.Data.external_dir)
        #

    if cfg.Data.dataset_name == 'BagDataset':
        from datasets.BagDataset import BagDataset


        dataset = BagDataset(df, **cfg.Data)
    elif cfg.Data.dataset_name == 'TwoStreamBagDataset':
        from datasets.TwoStreamBagDataset import TwoChannelBagDataset
        dataset = TwoChannelBagDataset(df, **cfg.Data)
    else:
        raise NotImplementedError


    if dataset_name == 'train_set':
        if cfg.Train.balance:

            weights = dataset.get_balance_weight()
            dataloader = DataLoader(dataset, batch_size=None, sampler=WeightedRandomSampler(weights, len(weights)),
                                    num_workers=cfg.Train.num_worker)


        else:
            dataloader = DataLoader(dataset, batch_size=None, shuffle=True,
                                    num_workers=cfg.Train.num_worker)
    else:
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False,
                                    num_workers=cfg.Train.num_worker)

    return dataloader
