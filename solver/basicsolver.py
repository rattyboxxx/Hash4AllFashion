import os
import numpy as np
from time import time

import torch
import torch.nn as nn
from torch.nn.parallel import data_parallel
from tensorboardX import SummaryWriter

import utils


##TODO: Modify this
class BasicSolver(object):
    """Base class for model solver."""

    def __init__(self, param, net, train_loader, test_loader, logger):
        """Use param to initialize a Solver instance."""
        from torch.backends import cudnn

        cudnn.benchmark = True
        self.logger = logger
        self.param = param
        self.net = net
        self.num_users = self.net.param.num_users
        self.best_acc = -np.inf
        self.best_loss = np.inf
        # data loader
        self.loader = dict(train=train_loader, test=test_loader)
        self.iter = 0
        self.last_epoch = -1
        self.parallel, self.device = utils.get_device(param.gpus)
        self.init_optimizer(param.optim_param)
        self.init_tracking_writer(param.tracking_method)
        
    def init_optimizer(self, optim_param):
        """Init optimizer."""
        # Set the optimizer
        optimizer = utils.get_named_class(torch.optim)[optim_param.name]
        groups = optim_param.groups
        num_child = self.net.num_groups()
        if len(groups) == 1:
            groups = groups * num_child
        else:
            # num of groups should match the network
            assert num_child == len(
                groups
            ), """number of groups {},
            while size of children is {}""".format(
                len(groups), num_child
            )
        param_groups = []
        for child, param in zip(self.net.children(), groups):
            param_group = {"params": child.parameters()}
            param_group.update(param)
            param_groups.append(param_group)
        self.optimizer = optimizer(param_groups, **optim_param.grad_param)
        # Set learning rate policy
        enum_lr_policy = utils.get_named_class(torch.optim.lr_scheduler)
        lr_policy = enum_lr_policy[optim_param.lr_scheduler]
        self.ReduceLROnPlateau = optim_param.lr_scheduler == "ReduceLROnPlateau"
        self.lr_scheduler = lr_policy(self.optimizer, **optim_param.scheduler_param)

    def init_tracking_writer(self, tracking_method):
        """Init tracking method."""
        ##TODO: Code for visdom, comet, wandb
        assert tracking_method in ["tensorboard", "visdom", "comet", "wandb"]

        if tracking_method == "tensorboard":
            log_dir = self.param.tracking_log_dir
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            raise # not implemented yet
    
    def gather_loss(self, losses, backward=False):
        """Gather all loss according to loss weight defined in self.net.

        1. Each loss in losses is shape of batch size
        2. Weight that is None will not be added into final loss

        Requires
        --------
        The loss return by self.net.forward() should match
        net.loss_weight in keys.
        """
        loss = 0.0
        results = {}
        for name, value in losses.items():
            weight = self.net.loss_weight[name]
            value = value.mean()
            # Save the scale
            results[name] = value.item()
            if weight:
                loss += value * weight
        # Save overall loss
        results["loss"] = loss.item()
        # wether to do back-proba
        if backward:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return results

    @staticmethod
    def gather_accuracy(accuracy):
        """Gather all accuracy according in self.net.

        Each accuracy is shape of batch size. There no final accuracy.

        Return
        ------
        Each accuracy will be averaged and return.
        """
        return {k: v.sum().item() / v.numel() for k, v in accuracy.items()}

    def run(self):
        """Run solver from epoch [0, epochs - 1]."""
        while self.last_epoch < self.param.epochs - 1:
            self.step()
        ##TODO: close for different writer
        self.writer.close()

    def adjust_before_epoch(self, epoch):
        pass

    def step(self, epoch=None):
        """Step for train and evaluate.

        Re-implement `adjust_before_epoch` for extra setting.
        """
        if epoch is None:
            epoch = self.last_epoch + 1
        self.adjust_before_epoch(epoch)
        # Train and Test
        self.train_one_epoch(epoch)
        result = self.test_one_epoch(epoch)
        loss, acc = result["loss"], result["accuracy"]
        self.save(label="lastest")
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_acc = acc
            self.save_net(label="best")
            self.logger.info("Best model:")
            for k, v in result.items():
                self.logger.info("-------- %s: %.3f" % (k, v))
        if self.ReduceLROnPlateau:
            self.lr_scheduler.step(-loss, epoch=epoch)
        else:
            self.lr_scheduler.step(epoch)
        self.last_epoch = epoch
        return result

    def step_batch(self, inputs):
        """Compute one batch."""
        if self.parallel:
            return data_parallel(self.net, inputs, self.param.gpus)
        return self.net(*inputs)

    def train_one_epoch(self, epoch):
        """Run one epoch for training net."""
        # Update the epoch
        self.net.train()
        phase = "train"
        # Generate negative outfit before each epoch
        loader = self.loader[phase].make_nega()
        self.logger.info("\n".join(["\n", "="*10, "TRAINING", "="*10]))
        msg = "Train - Epoch[{}](%d): [%d]/[{}]:".format(epoch, loader.num_batch)
        lastest_time = time()
        ##TODO: Replace this with wandb, comet or tensorBoard
        tracer = utils.tracer.Tracer(win_size=0, logger=self.logger)
        self.net.rank_metric.reset()
        for idx, inputs in enumerate(loader):
            inputv = utils.to_device(inputs, self.device)
            batch_size = len(torch.unique(inputv[0]))
            data_time = time() - lastest_time
            loss_, accuracy_ = self.step_batch(inputv)
            loss = self.gather_loss(loss_, backward=True)
            accuracy = self.gather_accuracy(accuracy_)
            batch_time = time() - lastest_time
            results = dict(data_time=data_time, batch_time=batch_time)
            ##TODO: Change method for update accuracy and loss
            results.update(loss)
            results.update(accuracy)
            # update history and plotting
            tracer.update_history(x=self.iter, data=results, weight=batch_size)
            if self.iter % self.param.display == 0:
                self.logger.info(msg % (self.iter, idx))
                tracer.logging()
            self.iter += 1
            lastest_time = time()
        # Compute average results
        rank_results = self.net.rank_metric.rank()
        results = {k: v.avg for k, v in tracer.get_history().items()}
        results.update(rank_results)
        self.write_to_tracking(results, epoch, "train")

    def test_one_epoch(self, epoch=None):
        """Run test epoch.

        Parameter
        ---------
        epoch: current epoch. epoch = -1 means the test is
            done after net initialization.
        """
        if epoch is None:
            epoch = self.last_epoch
        # Test only use pair outfits
        self.net.eval()
        phase = "test"
        lastest_time = time()
        tracer = utils.tracer.Tracer(win_size=0, logger=self.logger)
        ##TODO: Do we need make_nega for testing phase, fix this
        loader = self.loader[phase].make_nega()
        num_batch = loader.num_batch
        self.logger.info("\n".join(["\n", "="*10, "TESTING", "="*10]))
        msg = "Epoch[{}]:Test [%d]/[{}]".format(epoch, num_batch)
        self.net.rank_metric.reset()
        test_iter = 0
        for idx, inputs in enumerate(loader):
            # Compute output and loss
            inputv = utils.to_device(inputs, self.device)
            batch_size = len(torch.unique(inputv[0]))
            data_time = time() - lastest_time
            with torch.no_grad():
                loss_, accuracy_ = self.step_batch(inputv)
                loss = self.gather_loss(loss_, backward=False)
                accuracy = self.gather_accuracy(accuracy_)
            # Update time and history
            batch_time = time() - lastest_time
            data = dict(data_time=data_time, batch_time=batch_time)
            ##TODO: Change method for update accuracy and loss
            data.update(loss)
            data.update(accuracy)
            # update history and plotting
            tracer.update_history(x=test_iter, data=data, weight=batch_size)
            if test_iter % self.param.test_display == 0:
                self.logger.info(msg % idx)
                tracer.logging()
            lastest_time = time()
            test_iter += 1
        # Compute average results
        rank_results = self.net.rank_metric.rank()
        results = {k: v.avg for k, v in tracer.get_history().items()}
        results.update(rank_results)
        self.write_to_tracking(results, epoch, "val")
        return results

    def write_to_tracking(self, results, epoch, prefix=""):
        if self.param.tracking_method == "tensorboard":
            for k, v in results.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, epoch)
        else:
            raise  # not implemented yet

    def save(self, label=None):
        if label is None:
            label = str(self.last_epoch)
        # self.save_solver(label)
        self.save_net(label)

    def save_solver(self, label):
        """Save the state of solver."""
        ##TODO: Should we modify this? 
        state_dict = {
            "net": self.net.state_dict(),
            "optim": self.optimizer.state_dict(),
            "state": dict(
                best_loss=self.best_loss,
                best_acc=self.best_acc,
                iter=self.iter,
                param=self.param,
                last_epoch=self.last_epoch,
            ),
            "scheduler": {
                k: v
                for k, v in self.lr_scheduler.__dict__.items()
                if k not in ["is_better", "optimizer"] and not callable(v)
            },
        }
        solver_path = self.format_filepath(label, "solver")
        self.logger.info("Save solver state to %s" % solver_path)
        torch.save(state_dict, solver_path)

    def save_net(self, label):
        """Save the net's state."""
        ##TODO: Should we split state_dict to sub modules state dict for later easy load checkpoints??
        model_path = self.format_filepath(label, "net")
        self.logger.info("Save net state to %s" % model_path)
        torch.save(self.net.state_dict(), model_path)

    def format_filepath(self, label, suffix):
        """Return file-path."""
        filename = self.param.checkpoints + "_" + label + "." + suffix
        return filename