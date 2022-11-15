import os
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule

from ..graph_modifier_base import GraphModifierBase

from tqdm import tqdm

def cantor_pairing(a):
    return a[1] + ((a[0] + a[1])*(a[0] + a[1] + 1))//2

def cantor_pairing_inv(z):
    def f(w):
        return (w*(w+1))//2

    def q(z):
        return np.floor(0.5*(np.sqrt(8*z + 1) - 1))

    res = np.zeros((2, len(z)), dtype=int)
    res[1] = z - f(q(z))
    res[0] = q(z) - res[1]

    return res

def print_eff_pur(event):
    cantor_true = cantor_pairing(event.modulewise_true_edges)
    cantor_pred = cantor_pairing(event.edge_index)
    cantor_intersection = np.intersect1d(cantor_true, cantor_pred)
    # print("bal", len(np.intersect1d(cantor_pred, cantor_true)))
    # print("len cantor true", len(cantor_true))
    # print("len cantor pred", len(cantor_pred))
    # print("len intersection", len(cantor_intersection))

    print("eff",len(cantor_intersection)/len(cantor_true),"pur",len(cantor_intersection)/len(cantor_pred))

class SimpleGraphModifier(GraphModifierBase):
    def __init__(self, hparams):
        super().__init__(hparams)

        print("Make graphs with purity {} and efficiency 1.0".format(self.target_pur))

        # Do short test at construction time
        test = np.random.randint(0,100,(2,10))
        assert (test == cantor_pairing_inv(cantor_pairing(test))).all()

    def prepare_data(self):
        for name, split in zip(self.datatypes, self.splits):
            this_input_dir = os.path.join(self.input_dir, name)
            assert os.path.exists(this_input_dir)

            this_output_dir = os.path.join(self.output_dir, name)
            os.makedirs(this_output_dir, exist_ok=True)

            all_events  = sorted(
                np.unique([os.path.join(this_input_dir, event[:14]) for event in os.listdir(this_input_dir)])
            )[:split]

            for event_file in tqdm(all_events, desc="modify {} graphs".format(name)):
                event = torch.load(event_file).detach().cpu()

                # print_eff_pur(event)

                cantor_true = cantor_pairing(event.modulewise_true_edges)
                cantor_pred = cantor_pairing(event.edge_index)

                # Make efficiency 1
                cantor_eff = np.union1d(cantor_true, cantor_pred)

                # Sort so that true edges are at beginning and false at end
                isin_idxs = np.argsort(np.logical_not(np.isin(cantor_eff, cantor_true)))
                cantor_eff = cantor_eff[isin_idxs]

                # Remove edges from the end so that purity equals target purity
                n_false = int(len(cantor_true)*(1 - self.target_pur) / self.target_pur)
                new_size = len(cantor_true) + n_false
                if new_size < len(cantor_eff):
                    cantor_eff = cantor_eff[:new_size]

                # create truth array
                y_true = torch.cat([torch.ones(len(cantor_true)), torch.zeros(n_false)])
                assert len(cantor_eff) == len(y_true)

                # shuffle true and false again
                shuffle_idxs = np.arange(len(cantor_eff))
                np.random.shuffle(shuffle_idxs)

                cantor_eff = cantor_eff[shuffle_idxs]
                y_true = y_true[shuffle_idxs]

                # set new edge index and true
                event.edge_index = torch.tensor(cantor_pairing_inv(cantor_eff))
                event.y = y_true

                # print_eff_pur(event)

                torch.save(event, os.path.join(this_output_dir, os.path.basename(event_file)))



