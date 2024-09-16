import torch
import torch.nn as nn
import torch.nn.init as torch_init
import random
import numpy as np
import random

class Memory(nn.Module):
    def __init__(self, dim,K):
        super().__init__()

        self.K=K

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue", torch.randn(dim, K))



    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):


        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        if (ptr + batch_size  > self.K):
            ptr = self.K - batch_size


        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer


        self.queue_ptr[0] = ptr



    @torch.no_grad()
    def _return_queue(self):
        batch_size = 128

        ids=random.sample(range(self.K),batch_size)

        top_p=self.queue[:,ids]

        top_p=top_p.T


        return top_p







