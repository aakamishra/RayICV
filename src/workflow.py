import torch
import torch.nn.functional as F


class Workflow:
    @staticmethod
    def train(model, device, train_loader, optimizer):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            train_loss += F.nll_loss(output, target, reduction='sum').item()
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader.dataset)
        return train_loss


    @staticmethod
    def val(model, device, val_loader):
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)

        return val_loss, accuracy
