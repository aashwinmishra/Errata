
import torch
import pytorch_lightning as L
from utils import MyAccuracy

class NN(L.LightningModule):
  def __init__(self,
               input_size: int=784,
               num_classes: int=10,
               loss: torch.nn.Module=torch.nn.CrossEntropyLoss()
                ):
    super().__init__()
    self.layer = torch.nn.Linear(input_size, num_classes)
    self.loss = loss
    self.accuracy = MyAccuracy()#torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)

  def forward(self, x):
    return self.layer(x)

  def training_step(self, batch, batch_idx):
    inputs, labels = batch
    inputs = inputs.view(inputs.shape[0], -1)
    logits = self.forward(inputs)
    loss = self.loss(logits, labels)
    acc = self.accuracy(logits, labels)
    self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    inputs, labels = batch
    inputs = inputs.view(inputs.shape[0], -1)
    logits = self.forward(inputs)
    loss = self.loss(logits, labels)
    acc = self.accuracy(logits, labels)
    self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True)
    

  def test_step(self, batch, batch_idx):
    inputs, labels = batch
    inputs = inputs.view(inputs.shape[0], -1)
    logits = self.forward(inputs)
    loss = self.loss(logits, labels)
    acc = self.accuracy(logits, labels)
    self.log_dict({"test_loss": loss, "test_acc": acc})
    return torch.argmax(logits, dim=-1)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

