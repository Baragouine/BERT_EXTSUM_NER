import random
import numpy as np
import torch

class DataLoader():
    def __init__(self, dataset, batch_size=1, shuffle=False, ner=False, prop=None):
        assert batch_size > 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lidx = list(range(len(self.dataset)))
        self.ner = ner
        self.prop = prop
        
        if self.prop is not None:
            assert self.prop >= 0
            assert self.prop <= 1

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

        if self.prop is not None:
            if (idx >= self.prop * (len(self.lidx) / self.batch_size)):
                return self.dataset[len(self.dataset)]

        if (idx >= len(self.lidx) / self.batch_size):
            return self.dataset[len(self.dataset)]

        idxs = self.lidx[idx*self.batch_size:idx*self.batch_size+self.batch_size]
        batch = [self.dataset[i] for i in idxs]

        batch = self.merge(batch)

        return batch

    def __len__(self):
        if self.prop is not None:
            return int(self.prop * len(self.lidx) / self.batch_size)

        return int(len(self.lidx) / self.batch_size)

    def merge(self, batch):
        list_input_ids = []
        list_attention_mask = []
        list_labels_sum = []
        list_labels_ner = []

        def merge_list(l):
            res = []
            for e in l:
                res = res + e
            return res

        idxs = [e["idx"] for e in batch]

        max_doc_len = max([len(e["input_ids"]) for e in batch])
        for block in range(max_doc_len):
            line = []

            for e in batch:
                if len(e["input_ids"]) > block:
                    line.append(e)

            list_input_ids.append(torch.cat([e["input_ids"][block].unsqueeze(0) for e in line], dim=0))
            list_attention_mask.append(torch.cat([e["attention_mask"][block].unsqueeze(0) for e in line]))
            list_labels_sum.append(merge_list([e["labels_sum"][block] for e in line]))
            list_labels_ner.append(merge_list([e["labels_ner"][block] for e in line]))
        
        #list_labels_sum = merge_list(list_labels_sum)
        #list_labels_ner = merge_list(list_labels_ner)

        return {"idx" : idxs, "input_ids" : list_input_ids, "attention_mask": list_attention_mask, "labels_sum" : list_labels_sum, "labels_ner" : list_labels_ner}
        