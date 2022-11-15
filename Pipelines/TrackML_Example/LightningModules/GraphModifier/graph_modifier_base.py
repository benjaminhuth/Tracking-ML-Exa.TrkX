# 3rd party imports
from pytorch_lightning import LightningDataModule


class GraphModifierBase(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.input_dir = self.hparams["input_dir"]
        self.output_dir = self.hparams["output_dir"]
        self.datatypes = self.hparams["datatype_names"]
        self.splits = self.hparams["datatype_split"]
        self.target_pur = self.hparams["target_purity"]
