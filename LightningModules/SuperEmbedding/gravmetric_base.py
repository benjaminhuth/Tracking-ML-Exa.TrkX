"""The base classes for the embedding process.

The embedding process here is both the embedding models (contained in Models/) and the training procedure, which is a Siamese network strategy. Here, the network is run on all points, and then pairs (or triplets) are formed by one of several strategies (e.g. random pairs (rp), hard negative mining (hnm)) upon which some sort of contrastive loss is applied. The default here is a hinge margin loss, but other loss functions can work, including cross entropy-style losses. Also available are triplet loss approaches.

Example:
    See Quickstart for a concrete example of this process.
    
Todo:
    * Refactor the training & validation pipeline, since the use of the different regimes (rp, hnm, etc.) looks very messy
"""

import logging

# 3rd party imports
import torch

# Local Imports
from ..Embedding.utils import build_edges
from ..Embedding.embedding_base import EmbeddingBase

device = "cuda" if torch.cuda.is_available() else "cpu"

class GravMetricBase(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        self.save_hyperparameters(hparams)

    def get_hinge_distance(self, spatial1, spatial2, e_spatial, y_cluster):

        hinge = y_cluster.float().to(self.device)
        hinge[hinge == 0] = -1

        reference = spatial1[e_spatial[0]]
        neighbors = spatial2[e_spatial[1]]
        d = torch.sum((reference - neighbors) ** 2, dim=-1)

        return hinge, d

    def build_training_set(self, batch, embedding_a, embedding_b, r_train=None, knn=None):

        # Instantiate empty prediction edge list
        training_edges = torch.empty([2, 0], dtype=torch.int64, device=self.device)

        query_indices, query = self.get_query_points(batch, embedding_a)
        
        # Append Hard Negative Mining (hnm) with KNN graph
        if "hnm" in self.hparams["regime"]:
            training_edges = self.append_hnm_pairs(training_edges, query, query_indices, embedding_b, r_train, knn)

        # Append random edges pairs (rp) for stability
        if "rp" in self.hparams["regime"]:
            training_edges = self.append_random_pairs(training_edges, query_indices, embedding_b)

        # Instantiate bidirectional truth (since KNN prediction will be bidirectional)
        e_bidir = batch.signal_true_edges

        # Calculate truth from intersection between Prediction graph and Truth graph
        training_edges, y = self.get_truth(batch, training_edges, e_bidir)
        new_weights = y.to(self.device)

        # Append all positive examples and their truth and weighting
        training_edges, y, new_weights = self.get_true_pairs(
            training_edges, y, new_weights, e_bidir
        )

        return training_edges, y

    def get_loss(self, hinge, d, margin=None, weight=1):

        if margin is None:
            margin = self.hparams["margin"]**2

        negative_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == -1],
            hinge[hinge == -1],
            margin=margin,
            reduction="mean",
        )

        positive_loss = torch.nn.functional.hinge_embedding_loss(
            d[hinge == 1],
            hinge[hinge == 1],
            margin=margin,
            reduction="mean",
        )

        loss = negative_loss +  weight * positive_loss

        return loss

    def training_step(self, batch, batch_idx):

        """
        Args:
            batch (``list``, required): A list of ``torch.tensor`` objects
            batch (``int``, required): The index of the batch

        Returns:
            ``torch.tensor`` The loss function as a tensor
        """

        logging.info(f"Memory at train start: {torch.cuda.max_memory_allocated() / 1024**3} Gb")

        # Forward pass of model, handling whether Cell Information (ci) is included
        input_data = self.get_input_data(batch)

        # Embed hits
        topo_a, topo_b, output_a, output_b = self(input_data)

        # Build training set
        e_spatial, y_spatial = self.build_training_set(batch, output_a, output_b)
        e_topo, y_topo = self.build_training_set(batch, topo_a, topo_b, r_train=self.hparams["topo_margin"])

        # Loss functions
        spatial_hinge, spatial_d = self.get_hinge_distance(output_a, output_b, e_spatial, y_spatial)
        topo_hinge, topo_d = self.get_hinge_distance(topo_a, topo_b, e_topo, y_topo)

        spatial_loss = self.get_loss(spatial_hinge, spatial_d, self.hparams["margin"]**2, self.hparams["weight"])
        topo_loss = self.get_loss(topo_hinge, topo_d, self.hparams["topo_margin"]**2, self.hparams["topo_weight"])
        loss = spatial_loss + topo_loss      

        self.log("train_loss", loss)

        return loss

    def shared_evaluation(self, batch, batch_idx, knn_radius, knn_num, log=False):

        input_data = self.get_input_data(batch)
        topo_a, topo_b, output_a, output_b = self(input_data)

        e_bidir = batch.signal_true_edges

        # Build whole KNN graph
        e_spatial = build_edges(
            output_a, output_b, indices=None, r_max=knn_radius, k_max=knn_num
        )
        e_topo = build_edges(
            topo_a, topo_b, indices=None, r_max=knn_radius, k_max=knn_num
        )

        e_spatial, y_cluster = self.get_truth(batch, e_spatial, e_bidir)
        e_topo, y_topo = self.get_truth(batch, e_topo, e_bidir)

        hinge, d = self.get_hinge_distance(
            output_a, output_b, e_spatial.to(self.device), y_cluster
        )

        loss = torch.nn.functional.hinge_embedding_loss(
            d, hinge, margin=self.hparams["margin"]**2, reduction="mean"
        )

        cluster_true = e_bidir.shape[1]
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        topo_true = e_bidir.shape[1]
        topo_true_positive = y_topo.sum()
        topo_positive = len(e_topo[0])

        eff = cluster_true_positive / cluster_true
        pur = cluster_true_positive / cluster_positive
        topo_eff = topo_true_positive / topo_true
        topo_pur = topo_true_positive / topo_positive

        if "module_veto" in self.hparams["regime"]:
            module_veto_pur = cluster_true_positive / (batch.modules[e_spatial[0]] != batch.modules[e_spatial[1]]).sum()
        else:
            module_veto_pur = 0
        
        if log:
            current_lr = self.optimizers().param_groups[0]["lr"]
            self.log_dict(
                {"val_loss": loss, "eff": eff, "pur": pur, "module_veto_pur": module_veto_pur, "current_lr": current_lr, "topo_eff": topo_eff, "topo_pur": topo_pur}
            )
        logging.info("Efficiency: {}".format(eff))
        logging.info("Purity: {}".format(pur))
        logging.info(batch.event_file)

        return {
            "loss": loss,
            "distances": d,
            "preds": e_spatial,
            "truth": y_cluster,
            "truth_graph": e_bidir,
        }