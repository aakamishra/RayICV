from sklearn.model_selection import KFold
import torch


class CrossValidationFoldGenerator:
    def __init__(self, dataset, folds, batch_size, shuffle=True):
        self.dataset = dataset
        self.folds = folds
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __repr__(self):
        return f'CrossValidationFoldGenerator(dataset={self.dataset}, \
            folds={self.folds}, batch_size={self.batch_size}, \
            shuffle={self.shuffle})'

    def generate(self):
        # create fold generator
        kfold = KFold(n_splits=self.folds,shuffle=self.shuffle)

        # partitions output var
        partitions = {}
        for fold,(train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            # sample from the folds given the indics for the fold split
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_loader = torch.utils.data.DataLoader(self.dataset,
                batch_size=self.batch_size, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(self.dataset,
                batch_size=self.batch_size, sampler=val_subsampler)
            partitions[fold] = (train_loader, val_loader)
        return partitions



