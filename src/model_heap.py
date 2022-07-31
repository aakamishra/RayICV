import heapq


class ModelHeap:
    def __init__(self, model_indices):
        self.heap = []
        for i in model_indices:
            self.heap.append((0, 0, i))
        # Python's heapq library implements a min heap
        heapq.heapify(self.heap)

    def push_model(self, model, folds, acc):
        return heapq.heappush(self.heap, (-folds, acc, model))

    def pop_model(self):
        return heapq.heappop(self.heap)

    def pushpop_models(self, model, folds, acc):
        return heapq.heappushpop(self.heap, (-folds, acc, model))

    def __str__(self):
        return str([(-inv_folds, acc, i) for (inv_folds, acc, i) in self.heap])
