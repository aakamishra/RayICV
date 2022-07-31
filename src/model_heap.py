import heapq


class ModelHeap:
    def __init__(self, model_indices):
        self.heap = []
        for i in model_indices:
            self.heap.append((0, 0, 0, i))
        # Python's heapq library implements a min heap
        heapq.heapify(self.heap)

    def push_model(self, model, folds, train_loss, val_loss):
        return heapq.heappush(self.heap, (-folds, -val_loss, -train_loss, model))

    def pop_model(self):
        return heapq.heappop(self.heap)

    def pushpop_models(self, model, folds, train_loss, val_loss):
        return heapq.heappushpop(self.heap, (-folds, -val_loss, -train_loss, model))

    def __str__(self):
        return str([(-inv_folds, -inv_val_loss, -inv_train_loss, i) for (inv_folds, inv_val_loss, inv_train_loss, i) in self.heap])
