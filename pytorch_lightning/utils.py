
import torch
import pytorch_lightning as L
import torchmetrics
from torchmetrics import Metric


class MyAccuracy(Metric):
  def __init__(self):
    super().__init__()
    self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds, target):
    #preds, target = self._input_format(preds, target)
    self.correct += torch.sum(torch.argmax(preds, dim=-1) == target)
    self.total += target.numel()

  def compute(self):
    return self.correct.float() / self.total
