import torch

from matplotlib import pyplot as plt
from stopping_criteria import LRFinderStoppingCriteria


class LRFinder():
    def __init__(self, learner):
        self.results = []
        self.learner = learner

    def find(self, lr_start=1e-6, lr_multiplier=1.1, smoothing=0.3):
        torch.save(self.learner.net.state_dict(), "params.pth")
        torch.save(self.learner.trainer.state_dict(), "state.pth")

        lr = lr_start

        stop_criteria = LRFinderStoppingCriteria(smoothing)

        while True:
            loss = self.learner.iteration(lr)
            self.results.append((lr, loss))
            if stop_criteria(loss):
                break
            lr = lr * lr_multiplier

        self.learner.net.load_state_dict(torch.load("params.pth"))
        self.learner.trainer.load_state_dict(torch.load("state.pth"))

        return self.results

    def plot(self):

        lrs = [e[0] for e in self.results]
        losses = [e[1] for e in self.results]
        plt.figure(figsize=(6, 8))
        plt.scatter(lrs, losses)
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        axes = plt.gca()
        axes.set_xlim(lrs[0], lrs[-1])
        y_lower = min(losses) * 0.8
        y_upper = losses[0] * 4
        axes.set_ylim([y_lower, y_upper])
        plt.show()
