import torch
import torch.nn.functional as F


class Workflow:
    @staticmethod
    def train(model, device, train_loader, optimizer):
        """Train Workflow"""

        # init model
        model.train()
        train_loss = 0

        # iterate over train loader
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # get gradient
            optimizer.zero_grad()
            output = model(data)

            # calculate metrics
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # get overall metric
        train_loss /= len(train_loader.dataset)
        return train_loss

    @staticmethod
    def val(model, device, val_loader):
        """Validaton Workflow"""

        # run evaluation method for the model
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            # do not train on the data
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # sum up batch loss
                val_loss += F.nll_loss(output, target, reduction='sum').item()
                # get the index of the max log-probability
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # get overall metrics
        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)

        return val_loss, accuracy
