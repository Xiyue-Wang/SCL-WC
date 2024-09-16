import os

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score


def train_loop(epoch, model, loader, optimizer, writer, cfg, **kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss = 0
    model.to(device)

    with tqdm(total=len(loader), desc='train epoch: {}'.format(epoch)) as bar:
        for idx, batch in enumerate(loader):
            x, y = batch['x'].to(device, dtype=torch.float32), \
                   batch['y'].to(device, dtype=torch.long)
            result = model(x, label=y, instance_eval=True) # (1, n_classes)
            inst_loss = result['inst_loss']
            bag_logits = result['bag_logits']
            contra_loss = result['contra_loss']
            bag_loss = loss_fn(bag_logits, y)

            loss = 0.8 * bag_loss + 0.2 * inst_loss +0.01 * contra_loss


            bar.set_postfix({'loss': '{:.5f}'.format(loss)})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            bar.update(1)
            if writer:
                writer.add_scalar('train/loss', loss, epoch * len(loader) + idx)
    train_loss /= len(loader)


