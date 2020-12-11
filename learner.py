import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


class Learner():
    def __init__(self, net, data_loader, device):
        self.iteration_loss = 0.0
        self.net = net
        self.data_loader = data_loader
        self.device = device

        self.data_loader_iter = iter(self.data_loader)
        self.loss_fn = CrossEntropyLoss()
        self.trainer = Adam(net.parameters(), lr=3e-4)

    def iteration(self, lr=None, take_step=True):
        if lr and (lr != self.trainer.param_groups[0]["lr"]):
            for group in self.trainer.param_groups:
                group['lr'] = lr

        data, label = next(self.data_loader_iter)
        data = data.to(self.device)
        label = label.to(self.device)
        self.trainer.zero_grad()

        output = self.net(data)
        loss = self.loss_fn(output, label)
        loss.backward()

        if take_step:
            self.trainer.step()

        self.iteration_loss = torch.mean(loss)

        return self.iteration_loss.cpu().detach().numpy()

    def close(self):
        self.data_loader_iter.shutdown()
