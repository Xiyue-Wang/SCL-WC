import argparse
import os
import shutil

import torch
from tensorboardX import SummaryWriter

from utils.utils import read_yaml, seed_torch, EarlyStopping, load_task_state, save_task_state
from utils.model_factory import load_model
from utils.dataloader_factory import create_dataloader
from utils.optimizer_factory import create_optimizer
from utils.training_method_factory import create_training_loop, create_validation

def training_task(index, result_dir, cfg):
    model = load_model(cfg)
    train_loader = create_dataloader(index, 'train_set', cfg, result_dir)
    val_loader = create_dataloader(index, 'val_set', cfg, result_dir)

    print('#'*30 + '\n' 
        f'load dataset successfully!\n'
        f'train set size: {len(train_loader.dataset)}\n'
        f'val set size: {len(val_loader.dataset)}\n'
        + '#'*30
    )

    # tensorboardX writer
    writer_dir = os.path.join(result_dir, 'log', str(index))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir)
    writer = SummaryWriter(writer_dir, flush_secs=15)

    optimizer = create_optimizer(model, cfg)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.Train.CosineAnnealingLR.T_max,
        eta_min=cfg.Train.CosineAnnealingLR.eta_min,
        last_epoch=-1
    )

    early_stopping = EarlyStopping(
        patience=cfg.Train.Early_stopping.patient,
        stop_epoch=cfg.Train.Early_stopping.stop_epoch,
        type= 'min' if cfg.Train.Early_stopping.type in ['loss'] else 'max'
    )



    current_epoch = 0

    train_loop = create_training_loop(cfg)
    validation = create_validation(cfg)

    print('#'*10 + ' training task start! ' + '#' * 10)


    for epoch in range(current_epoch, cfg.Train.max_epochs):

        lr = scheduler.get_last_lr()[0]
        print('learning rate:{:.8f}'.format(lr))
        writer.add_scalar('train/lr', lr, epoch)

        # training
        train_loop(
            epoch=epoch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            writer=writer,
            cfg=cfg
        )

        # validation
        stop = validation(
            index=index,
            epoch=epoch,
            model=model,
            loader=val_loader,
            result_dir=result_dir,
            early_stopping=early_stopping,
            writer=writer,
            cfg=cfg
        )

        if stop:
            break
        scheduler.step()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/task0_sample.yaml')
    parser.add_argument('--begin', type=int, default=0)
    parser.add_argument('--end', type=int, default=5)
    parser.add_argument('--set_seed', action='store_true')
    args = parser.parse_args()

    cfg = read_yaml(args.config_path)
    result_dir = os.path.join(cfg.General.result_dir, args.config_path.split('/')[-1])
    os.makedirs(result_dir, exist_ok=True)
    if not os.path.exists(os.path.join(result_dir, args.config_path.split('/')[-1])):
        shutil.copy(args.config_path, result_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(args.begin, args.end):
        if args.set_seed:
            seed_torch(device, cfg.General.seed + i)
        training_task(i, result_dir, cfg)

