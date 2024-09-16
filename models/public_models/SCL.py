import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from utils.loss import loss_fg,loss_bg,SupConLoss
from utils.memory import Memory


class Att_Head(nn.Module):
    def __init__(self,FEATURE_DIM,ATT_IM_DIM):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x 1, N * D



class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x





class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="large", dropout=True, k_sample=5, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,pos_num=1024,neg_num=16384,hard_neg_num=1024,**kwargs):
        super(CLAM_SB, self).__init__()
        self.size_dict = {'xs': [384, 256, 256], "small": [768, 512, 256], "big": [1024, 512, 384], 'large': [2048, 1024, 512]}
        size = self.size_dict[size_arg]
        print(size)
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


        initialize_weights(self)

        self.att_head = Att_Head(size[1],size[2])
        self.pos_mem=Memory(size[1],pos_num)
        self.neg_mem = Memory(size[1], neg_num)

        self.hard_neg_mem = Memory(size[1], hard_neg_num)

        self.sup_loss=SupConLoss()

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier,i,epoch):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)

        if len(h) < 10 * self.k_sample:
            k = 3
        else:
            k = self.k_sample
        top_p_ids = torch.topk(A, k)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(k, device)
        n_targets = self.create_negative_targets(k, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        contra_loss=0.0
        if epoch>4:
            if i==0:
                easy_neg = nn.functional.normalize(self.neg_mem._return_queue(), dim=1)
                hard_neg= nn.functional.normalize(self.hard_neg_mem._return_queue(),dim=1)
                pos_sample= nn.functional.normalize(self.pos_mem._return_queue(),dim=1)

                contra_pos_label = self.create_positive_targets(pos_sample.shape[0], device)
                contra_hard_neg_label = self.create_negative_targets(hard_neg.shape[0], device)
                contra_easy_neg_label = self.create_negative_targets(easy_neg.shape[0], device)
                contra_pos_hard_label=torch.cat([contra_pos_label,contra_hard_neg_label])
                contra_pos_hard_fea=torch.cat([pos_sample,hard_neg],dim=0).unsqueeze(dim=1)



                contra_pos_easy_label=torch.cat([contra_pos_label,contra_easy_neg_label])
                contra_pos_easy_fea=torch.cat([pos_sample,easy_neg],dim=0).unsqueeze(dim=1)
                contra_loss=contra_loss+self.sup_loss(contra_pos_hard_fea,contra_pos_hard_label)+self.sup_loss(contra_pos_easy_fea,contra_pos_easy_label)

                self.hard_neg_mem._dequeue_and_enqueue(top_p)

            elif i==1:

                easy_neg = nn.functional.normalize(self.neg_mem._return_queue(), dim=1)
                hard_neg = nn.functional.normalize(self.hard_neg_mem._return_queue(), dim=1)
                pos_sample = nn.functional.normalize(self.pos_mem._return_queue(), dim=1)
                contra_pos_label = self.create_positive_targets(pos_sample.shape[0], device)
                contra_hard_neg_label = self.create_negative_targets(hard_neg.shape[0], device)
                contra_easy_neg_label = self.create_negative_targets(easy_neg.shape[0], device)

                contra_pos_hard_label = torch.cat([contra_pos_label, contra_hard_neg_label])
                contra_pos_hard_fea = torch.cat([pos_sample, hard_neg], dim=0).unsqueeze(dim=1)

                contra_pos_easy_label = torch.cat([contra_pos_label, contra_easy_neg_label])
                contra_pos_easy_fea = torch.cat([pos_sample, easy_neg], dim=0).unsqueeze(dim=1)

                contra_loss = contra_loss+self.sup_loss(contra_pos_hard_fea, contra_pos_hard_label) + self.sup_loss(
                    contra_pos_easy_fea, contra_pos_easy_label)


                self.pos_mem._dequeue_and_enqueue(top_p)
            else:
                print(0)
        else:

            if i == 0:
                self.hard_neg_mem._dequeue_and_enqueue(top_p)
            elif i == 1:

                self.pos_mem._dequeue_and_enqueue(top_p)
            else:
                print(1)
        return instance_loss, all_preds, all_targets,contra_loss

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def forward(self, h, epoch=0,label=None, instance_eval=False, return_features=False, attention_only=False,):
        device = h.device
        h_raw=h
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N


        if instance_eval:

            one_hot_label= F.one_hot(label, num_classes=self.n_classes).squeeze()
        if one_hot_label[0].item()==1:

            self.neg_mem._dequeue_and_enqueue(h)
        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets,contra_loss = self.inst_eval(A, h, classifier,i,epoch)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)  # A: 1 * N h: N * 512 => M: 1 * 512
        logits = self.classifiers(M)  # 1 * K
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        result = {
            'bag_logits': logits,
            'attention_raw': A_raw,
            'M': M
        }

        if instance_eval:
            result['contra_loss']=contra_loss
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
            result['inst_loss'] = total_inst_loss
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return result


class SCL(CLAM_SB):
    def __init__(self, gate = True, size_arg = "small", dropout = True, k_sample=5, n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False,pos_num=2048,neg_num=16384,hard_neg_num=2048,**kwargs):
        nn.Module.__init__(self)
        self.size_dict = {'xs': [384, 328, 256], "small": [768, 512, 256], "big": [512, 384, 256], 'large': [1024, 768, 512]}
        size = self.size_dict[size_arg]
        print(size)
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_classes = n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in range(n_classes)] #use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn.cuda()
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)
        self.att_head = Att_Head(size[1],size[2])
        self.pos_mem=Memory(size[1],pos_num)
        self.neg_mem = Memory(size[1], neg_num)
        self.hard_neg_mem = Memory(size[1], hard_neg_num)
        self.sup_loss=SupConLoss()



    def forward(self, h, epoch=0,label=None, instance_eval=False, return_features=False, attention_only=False):
        device = h.device
        h_raw = h
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:

            one_hot_label= F.one_hot(label, num_classes=self.n_classes).squeeze()

            if one_hot_label[0].item()==1:
                self.neg_mem._dequeue_and_enqueue(h)

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze() #binarize label

            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1: #in-the-class:
                    instance_loss, preds, targets,contra_loss = self.inst_eval(A[i], h, classifier,i,epoch)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else: #out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        result = {
            'bag_logits': logits,
            'attention_raw': A_raw,
            'M': M
        }
        if instance_eval:
            result['contra_loss']=contra_loss
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
            'inst_preds': np.array(all_preds)}
            result['inst_loss'] = total_inst_loss
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return result


