import torch

from .basicsolver import BasicSolver


class FashionNetSolver(BasicSolver):
    
    def adjust_before_epoch(self, epoch):
        if self.param.increase_hard:
            prob = pow(epoch / self.param.epochs, 0.5)
            self.loader["train"].set_prob_hard(prob)
        scale = pow(epoch * self.param.gamma + 1, 0.5)
        self.net.set_scale(scale)