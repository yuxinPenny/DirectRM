import torch
from torch.utils.data import Dataset

class denovoDataset(Dataset):
    def __init__(self,seq,stat,bse,labels=None):
        if labels is None:
            self.labels = labels
        else:
            self.labels = torch.LongTensor(labels)
        self.seq = torch.from_numpy(seq)
        self.stat = torch.from_numpy(stat)
        self.bse = torch.from_numpy(bse)

    def __getitem__(self,id):
        if self.labels is None:
            return self.seq[id],self.stat[id],self.bse[id]
        else:
            return self.seq[id],self.stat[id],self.bse[id],self.labels[id]

    def __len__(self):
        return len(self.seq)

class pmlDataset(Dataset):
    def __init__(self,seq,stat,bse,labels=None,pseudo_labels=None):
        if labels is None:
            self.labels = labels
            self.pseudo_labels = pseudo_labels
        elif labels is not None and pseudo_labels is None:
            self.labels = torch.from_numpy(labels)
            self.pseudo_labels = pseudo_labels
        else:
            self.labels = torch.from_numpy(labels)
            self.pseudo_labels = torch.from_numpy(pseudo_labels)
        self.seq = torch.from_numpy(seq)
        self.stat = torch.from_numpy(stat)
        self.bse = torch.from_numpy(bse)

    def __getitem__(self,id):
        if self.labels is None:
            return self.seq[id],self.stat[id],self.bse[id]
        elif self.labels is not None and self.pseudo_labels is None:
            return self.seq[id],self.stat[id],self.bse[id],self.labels[id]
        else:
            return self.seq[id],self.stat[id],self.bse[id],self.labels[id],self.pseudo_labels[id]
    def __len__(self):
        return len(self.seq)