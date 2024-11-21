import torch
import pytorch_lightning as L 
import torchmetrics 
from models import NN 
from dataset import MnistDataModule 
from callbacks import MyPrintingCallback
from pytorch_lightning.loggers import TensorBoardLogger


logger = TensorBoardLogger(save_dir="logs", name="mnist_v0")
model = NN()
mnist = MnistDataModule()
trainer = L.Trainer(max_epochs=3, callbacks=[MyPrintingCallback()], logger=logger, profiler="simple")
trainer.fit(model=model, datamodule=mnist)
