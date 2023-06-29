import os
import numpy as np
import random
import torch

from torchsummary import summary
from torch import optim
from tqdm import tqdm

import monai

import utils
from optimizer import SGHM

from torch_loader import train_data_loader
from matrix import DiceCoef
from unet3D import Unet3D
import operations as ops

from torch import nn

torch.cuda.empty_cache()

device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

OPTIMIZATION_PARAMS = {
    'base_lr': 1e-4,
    'epochs': 200,
    'decay_step': 100,
    'gamma': 0.5,
    'checkpoint': 50,
    'criterion': 'CE',
    'scheduler': None,
    'optimizer': 'SGHM',
    'grad_clip': False,
    "matrix": "dice",
    "train_item": 180,
    "batch_size": 2,
    "epoch_noise": 45,
    'weight_decay': 0.0001,
    'temp': 0.50
}


class Trainer(object):
    def __init__(self, model, params=None, param_list=None, model_name='default', experiment=None):
        self.model = model.to(device)
        self.param_list = param_list or self.model.parameters()
        self.params = dict(OPTIMIZATION_PARAMS, **params) if params else OPTIMIZATION_PARAMS
        self.criterion, self.optimizer, self.matrix = self.set_optimization()
        self.model_path = utils.create_dirs(model_name, experiment)

    def set_optimization(self):

        if self.params['criterion'] == 'CE':
            criterion = nn.CrossEntropyLoss()

        elif self.params['criterion'] == 'dice_loss':
            criterion = monai.losses.DiceLoss(sigmoid=True)
        else:
            raise NotImplementedError("Please add the new loss ...")

        if self.params['optimizer'] == 'ADAM':
            optimizer = optim.Adam(self.param_list, lr=self.params['base_lr'],
                                   weight_decay=self.params['weight_decay'])

        elif self.params['optimizer'] == 'SGDM':
            optimizer = optim.SGD(self.param_list, lr=self.params['base_lr'], momentum=0.99,
                                  weight_decay=self.params['weight_decay'])

        elif self.params["optimizer"] == "SGHM":
            optimizer = SGHM(self.param_list, lr=self.params['base_lr'], momentum=0.99, dampening=0.0,
                             weight_decay=self.params["weight_decay"], N_train=self.params["train_item"],
                             temp=self.params["temp"], addnoise=True, epoch_noise=False)

        else:
            raise NotImplementedError("Please add the new optimizer ...")

        if self.params["matrix"] == "dice":
            matrix = DiceCoef()
        else:
            raise NotImplementedError("Please add the new matrix ...")
        return criterion, optimizer, matrix

    def train_step(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)  # set tensors in GPU

        gt = targets.argmax(1)

        self.optimizer.zero_grad()  # 1

        output = self.model(inputs)
        pred = output.argmax(1)

        target = gt.long()
        loss = self.criterion(output, target)

        loss.backward()  # 2

        if self.params['grad_clip']:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

        self.optimizer.step()  # 3

        summary = {
            'Mode': 'Train',
            'Loss': loss.item(),
            "Dice": self.matrix(pred, gt).item()
        }
        return summary

    @torch.no_grad()
    def val_step(self, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        gt = targets.argmax(1)

        output = self.model(inputs)
        pred = output.argmax(1)

        target = gt.long()
        loss = self.criterion(output, target)

        summary = {
            'Mode': 'Valid',
            'Loss': loss.item(),
            'Dice': self.matrix(pred, gt).item(),
        }
        return summary

    @staticmethod
    def get_summaries():
        losses = dict(train_loss=[], val_loss=[])
        metrics = dict(train_dice=[], val_dice=[])
        return losses, metrics

    def get_learning_rate(self):
        if self.params['scheduler'] == 'exponential_decay':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=self.params['decay_step'],
                                                        gamma=self.params['gamma'])
        elif self.params['scheduler'] == 'cyclical':
            scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,
                                                          self.params['base_lr'],
                                                          max_lr=self.params.get('max_lr', 1.0),
                                                          step_size_up=self.params.get('step_up',
                                                                                       1),
                                                          step_size_down=self.params.get(
                                                              'step_down', 1))
        elif self.params['scheduler'] == 'WarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                             T_0=self.params["decay_step"],
                                                                             T_mult=1)

        else:
            scheduler = None
        return scheduler

    def start_training(self, train_data, valid_data):

        # create global summaries
        loss_summary, metrics_summary = self.get_summaries()

        scheduler = self.get_learning_rate()

        for epoch in tqdm(range(1, self.params['epochs'])):
            print('Staring epoch: {}'.format(epoch))
            print('Learning rate:', scheduler.get_last_lr() if self.params['scheduler'] else self.params['base_lr'])

            step = 0
            # for batch in train_data:
            for batch_idx, batch in enumerate(train_data):
                inputs, labels = batch
                inputs, targets = inputs.to(device), labels.to(device)  # set tensors in GPU

                gt = targets.argmax(1)

                self.optimizer.zero_grad()  # 1

                output = self.model(inputs)
                pred = output.argmax(1)

                target = gt.long()
                loss = self.criterion(output, target)

                if (epoch % self.params['checkpoint']) + 1 > self.params["epoch_noise"] or (
                        epoch % self.params['checkpoint']) == 0:
                    self.optimizer.param_groups[0]['epoch_noise'] = True

                else:
                    self.optimizer.param_groups[0]['epoch_noise'] = False

                loss.backward()  # 2

                if self.params['grad_clip']:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.25)

                self.optimizer.step()  # 3

                train_summary = {
                    'Mode': 'Train',
                    'Loss': loss.item(),
                    "Dice": self.matrix(pred, gt).item()
                }
                # log metrics and loss
                loss_summary['train_loss'].append(train_summary['Loss'])
                metrics_summary['train_dice'].append(train_summary['Dice'])

                step += 1

            print('Validating ...')

            val_step = 0
            # for val_batch in valid_data:

            for i, val_batch in enumerate(valid_data):
                val_inputs, val_labels = val_batch
                val_summary = self.val_step(val_inputs, val_labels)
                # log metrics and loss for validation
                loss_summary['val_loss'].append(val_summary['Loss'])
                metrics_summary['val_dice'].append(val_summary['Dice'])

                val_step += 1

            average_losses = utils.compute_averages(loss_summary, step)
            average_metrics = utils.compute_averages(metrics_summary, step)
            template = 'Epoch: {}, train loss: {}, val loss: {}, train dice: {}, val dice: {}'
            print(template.format(epoch, *average_losses, *average_metrics))

            if scheduler is not None:
                scheduler.step()

            if (epoch % self.params['checkpoint']) == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.model_path, 'checkpoint_{}.pt'.format(epoch)))

        return loss_summary, metrics_summary


def train_model(model_name, experiment, epochs=2501, base_lr=0.1,
                weight_decay=0.00001, temp=0.50, loss_type='CE', train_item=199,
                optimizer='SGHM', init_type='normal', schedule_type='WarmRestarts',
                checkpoint=100, epoch_noise=90, grad_clip=False, decay_step=100):
    seed = 86
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    params = {
        'epochs': epochs,
        'base_lr': base_lr,
        'optimizer': optimizer,
        'scheduler': schedule_type,
        'criterion': loss_type,
        'grad_clip': grad_clip,
        'checkpoint': checkpoint,
        "epoch_noise": epoch_noise,
        'decay_step': decay_step,
        'train_item': train_item,
        'weight_decay': weight_decay,
        'temp': temp
    }
    train, valid = train_data_loader()

    model = Unet3D(n_channels=1, n_classes=3, n_filters=32, drop=None, bilinear=True)
    model.to(device)
    summary(model, (1, 128, 128, 128))
    ops.init_weights(model, init_type=init_type)
    trainer = Trainer(model,
                      params=params,
                      model_name=model_name,
                      experiment=experiment,
                      )
    loss_summary, metrics_summary = trainer.start_training(train, valid)
    torch.save(trainer.model.state_dict(), os.path.join(trainer.model_path, 'final.pt'))
    utils.save(os.path.join(trainer.model_path, 'losses.pz'), loss_summary)
    utils.save(os.path.join(trainer.model_path, 'metrics.pz'), metrics_summary)


if __name__ == '__main__':
    train_model(experiment='model_5',
                model_name='mcmc',
                epochs=2001,
                base_lr=0.1,
                checkpoint=50,
                epoch_noise=40,
                decay_step=50,
                train_item=199,
                optimizer="SGHM",
                weight_decay=0.00001,
                temp=0.50,
                init_type='normal',
                schedule_type='WarmRestarts')  # change experiment not model name
