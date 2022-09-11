import heapq


class ModelHeap:
    """Class for dealing with scheduling of model configurations."""

    def __init__(self, model_indices):
        """
        Intializes internal heap list.

        Params
        ------
        model_indices: number of models configurations to keep track of.

        Returns
        -------
        return: ModelHeap Object
        """

        # internal tracking list
        self.heap = []
        for i in model_indices:
            self.heap.append((0, 0, 0, i))
        # Python's heapq library implements a min heap
        heapq.heapify(self.heap)

    def push_model(self, model, folds, train_loss, val_loss):
        """Function for pushing back conf. w/ metrics"""
        return heapq.heappush(self.heap, (folds, val_loss, train_loss, model))

    def pop_model(self):
        """Function for popping top conf. w/ metrics"""
        return heapq.heappop(self.heap)

    def pushpop_models(self, model, folds, train_loss, val_loss):
        """Generic Push/Pop for Model Conf. Tuple"""
        return heapq.heappushpop(
            self.heap,
            (folds, val_loss, train_loss, model)
        )

    def folds_trained(self, folds):
        """Gets the number of models trained for a
        specified number of folds

        Params
        ------
        folds: number of folds the user wants to check

        Returns
        -------
        suma: the sum total of the number of models w/
        folds trained specified
        """
        suma = 0
        for i in range(len(self.heap)):
            if self.heap[i][0] == folds:
                suma += 1
        return suma

    def __str__(self):
        """Gives string representation of ModelHeap"""
        return str([
            (folds, val_loss, train_loss, i)
            for (folds, val_loss, train_loss, i) in self.heap
        ])
