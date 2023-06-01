import random
import numpy as np
import torch

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False, ner=False):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lidx = list(range(len(self.dataset)))
        self.ner = ner

        # Padding last batch if necessary
        if len(self.lidx) % self.batch_size != 0:
            self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

        # Shuffle if necessary
        if self.shuffle:
            random.shuffle(self.lidx)

    def __getitem__(self, idx):
        assert idx >= 0
        if idx == 0:
            self.lidx = list(range(len(self.dataset)))

            # Padding last batch if necessary
            if len(self.lidx) % self.batch_size != 0:
                self.lidx = self.lidx + random.sample(self.lidx, self.batch_size - (len(self.lidx) % self.batch_size))

            # Shuffle if necessary
            if self.shuffle:
                random.shuffle(self.lidx)
        if (idx >= len(self.lidx) / self.batch_size):
            return self.dataset[len(self.dataset)]
        idxs = self.lidx[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        batch = [self.dataset[i] for i in idxs]

        batch = self.merge(batch)

        return batch

    def __len__(self):
        return int(len(self.lidx) / self.batch_size)

    def merge(self, batch):
        def merge_list(l):
            res = []
            for e in l:
                res = res + e
            return res

        idxs = [e["idx"] for e in batch]
        input_ids = torch.cat([e["input_ids"].unsqueeze(0) for e in batch], dim=0)
        attention_mask = torch.cat([e["attention_mask"].unsqueeze(0) for e in batch])
        labels_sum = merge_list([e["labels_sum"] for e in batch])
        labels_ner = merge_list([e["labels_ner"] for e in batch])
        return {"idx" : idxs, "input_ids" : input_ids, "attention_mask": attention_mask, "labels_sum" : labels_sum, "labels_ner" : labels_ner}
        