from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

import torch.nn as nn

from torchmtlr import mtlr_neg_log_likelihood, mtlr_survival, mtlr_risk

from lifelines.utils import concordance_index

from src.models.modules.net import Image_MTLR, Dual_MTLR, EHR_MTLR


class DEEP_MTLR(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
       
        **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()


        if self.hparams['model'] == 'Dual':
            self.model = Dual_MTLR(hparams = self.hparams)

        else:
            print('Please select the correct model architecture name.')

        self.apply(self.init_params)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def init_params(self, m: torch.nn.Module):
        """Initialize the parameters of a module.
        Parameters
        ----------
        m
            The module to initialize.
        Notes
        -----
        Convolutional layer weights are initialized from a normal distribution
        as described in [1]_ in `fan_in` mode. The final layer bias is
        initialized so that the expected predicted probability accounts for
        the class imbalance at initialization.
        References
        ----------
        .. [1] K. He et al. ‘Delving Deep into Rectifiers: Surpassing
           Human-Level Performance on ImageNet Classification’,
           arXiv:1502.01852 [cs], Feb. 2015.
        """

        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, a=.1)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            # initialize the final bias so that the predictied probability at
            # init is equal to the proportion of positive samples
            nn.init.constant_(m.bias, -1.5214691)

    def step(self, batch: Any):
        x, y, labels = batch
        logits = self.forward(x)
        loss = mtlr_neg_log_likelihood(logits, y.float(), self.model, self.hparams['C1'], average=True)
        
        return loss, logits, y, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, labels = self.step(batch)

        # log train metrics
        #acc = self.train_accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        #self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, y, labels = self.step(batch)

        # log val metrics
        # acc = self.val_accuracy(preds, targets)
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "y": y, "labels": labels}

    def validation_epoch_end(self, outputs: List[Any]):
        
        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob   = torch.cat([x["preds"] for x in outputs]).cpu()          
        y           = torch.cat([x["y"] for x in outputs]).cpu()

        true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()


        pred_risk = mtlr_risk(pred_prob).detach().numpy()        
        ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"val/loss": loss,
               "val/ci": ci_event,
               }
        
        self.log_dict(log)
        return {"loss": loss, "CI": ci_event}



    def test_step(self, batch: Any, batch_idx: int):
        

        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs: List[Any]):

        """
        The CoxPH model by using lifelines or any other stable package can be used here to predict the risk score 
        based on the EHR records and then combined with the predictions of the Deep-MTLR net for final risk predictions. 
        """

        loss        = torch.stack([x["loss"] for x in outputs]).mean()
        pred_prob   = torch.cat([x["preds"] for x in outputs]).cpu()          
        y           = torch.cat([x["y"] for x in outputs]).cpu()

        true_time   = torch.cat([x["labels"]["time"] for x in outputs]).cpu()
        true_event  = torch.cat([x["labels"]["event"] for x in outputs]).cpu()


        pred_risk = mtlr_risk(pred_prob).detach().numpy()        
        ci_event  = concordance_index(true_time, -pred_risk, event_observed=true_event)

        log = {"test/loss": loss,
               "test/ci": ci_event,
               }
        
        self.log_dict(log)
        return 

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = Adam(self.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        scheduler = {
            "scheduler": MultiStepLR(optimizer, milestones=[50, 150]),
            "monitor": "loss",
        }
        return [optimizer], [scheduler]
