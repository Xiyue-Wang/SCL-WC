import os

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def train_loop(epoch, model, loader, optimizer, writer, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = 0
    model.to(device)

    with tqdm(total=len(loader), desc='train epoch: {}'.format(epoch)) as bar:
        for idx, batch in enumerate(loader):
            x, y = batch['x'].to(device, dtype=torch.float32), \
                                               batch['y'].to(device, dtype=torch.long)

            result = model(x) # (B, n_classes)
            logits = result[cfg.Model.logits_field]
            loss = loss_fn(logits, y)

            bar.set_postfix({'loss' : '{:.5f}'.format(loss)})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.update(1)

            if writer:
                writer.add_scalar('train/loss', loss, epoch * len(loader) + idx)
    train_loss /= len(loader)


def validation(index, epoch, model, loader, result_dir, early_stopping, writer, cfg):
    n_classes = cfg.Data.n_classes
    early_stopping_type = cfg.Train.Early_stopping.type

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader), desc='validate epoch:{}'.format(epoch)) as bar:
            for idx, batch in enumerate(loader):
                x, y = batch['x'].to(device, dtype=torch.float32), \
                       batch['y'].to(device, dtype=torch.long)
                result = model(x)
                logits = result[cfg.Model.logits_field]
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_fn(logits, y)
                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    val_loss /= len(loader)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    acc = accuracy_score(labels, np.argmax(probs, axis=-1).astype(np.int32))

    if n_classes == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = 0
        for i in range(n_classes):
            auc += roc_auc_score((labels == i).astype(np.int32), probs[:, i])
        auc /= n_classes

    print('\nVal Set, val_loss: {:.4f}, acc {:.4f}, auc: {:.4f}'.format(val_loss, acc, auc))

    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', acc, epoch)

    if early_stopping:
        os.makedirs(os.path.join(result_dir, 'best_checkpoints'), exist_ok=True)
        ckpt_name = os.path.join(result_dir, 'best_checkpoints', "s_{}_checkpoint.pt".format(index))
        if early_stopping_type == 'loss':
            early_stopping(epoch, val_loss, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'acc':
            early_stopping(epoch, acc, model, ckpt_name=ckpt_name)
        elif early_stopping_type == 'auc':
            early_stopping(epoch, auc, model, ckpt_name=ckpt_name)
        else:
            raise NotImplementedError
        if early_stopping.early_stop:
            print('Early stopping')
            return True

    return False

def evaluation(index, model, loader, result_dir, cfg):
    n_classes = cfg.Data.n_classes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0
    model.to(device)
    probs = []
    labels = []
    with torch.no_grad():
        with tqdm(total=len(loader)) as bar:
            for idx, batch in enumerate(loader):
                x, y = batch['x'].to(device, dtype=torch.float32), \
                       batch['y'].to(device, dtype=torch.long)
                result = model(x)
                logits = result[cfg.Model.logits_field]
                y_prob = torch.softmax(logits, dim=-1)
                loss = loss_fn(logits, y)

                probs.append(y_prob)
                labels.append(y)

                val_loss += loss.item()
                bar.update(1)

    labels = torch.cat(labels, dim=0).cpu().numpy()
    probs = torch.cat(probs, dim=0).cpu().numpy()
    id_list = loader.dataset.get_id_list()

    df_dict = {'id': id_list, 'label': labels}
    for i in range(n_classes):
        df_dict[f'prob_{i}'] = probs[:, i]
    pd.DataFrame(df_dict).to_csv(os.path.join(result_dir, f'preds_{index}.csv'), index=False)
    

