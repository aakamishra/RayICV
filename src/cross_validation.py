from sklearn.model_selection import KFold
import torch


class CrossValidationFoldGenerator:
    def __init__(self, dataset, folds, batch_size, shuffle=True):
        """
        Cross Validation Fold Generator creates the folds for
        the given dataset based on the given parameters.

        Params
        ------
        dataset: the user given dataset that has to be split into folds
        folds: the number of folds to partition
        batch_size: the batch length to init the subsampler method

        Returns
        -------
        partitions: dictionary of key value pairs between assigned folds
        and their respective train and validation data loaders

        """
        self.dataset = dataset
        self.folds = folds
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __repr__(self):
        """Return representation of class in the form of a string."""
        return f"CrossValidationFoldGenerator(dataset={self.dataset}, \
            folds={self.folds}, batch_size={self.batch_size}, \
            shuffle={self.shuffle})"

    def generate(self):
        """
        Creates a generated kfold object that iterates over the data while
        partitioning the dataset into the specified number of folds. This
        function also load the data into data loaders
        accordingly for each batch.
        """
        kfold = KFold(n_splits=self.folds, shuffle=self.shuffle)

        # partitions output var
        partitions = {}
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            # sample from the folds given the indics for the fold split
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

            # create the loader based on the information from the subsample
            train_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=val_subsampler)

            # load the configuration into the partition dictionary
            partitions[fold] = (train_loader, val_loader)
        return partitions
