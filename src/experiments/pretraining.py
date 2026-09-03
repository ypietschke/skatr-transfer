import torch

from src.experiments.training import TrainingExperiment
from src.utils.collators import JEPACollator


class PretrainingExperiment(TrainingExperiment):

    supervised: bool = False

    def get_collator(self, training):
        """Perform preprocessing and masking on CPU. Avoids GPU sync during training."""

        num_patches = tuple(
            s // p
            for s, p in zip(self.cfg.dataset.image_shape, self.cfg.net.patch_shape)
        )

        mask_cfg = self.cfg.masking

        # CHANGED 2026-05-13: forward self.augmentations and the `training` flag to
        # JEPACollator so that augmentations configured under cfg.training.augmentations
        # are actually applied (training split only). Previously augmentations were
        # loaded by init_augmentations() but silently dropped here.
        return JEPACollator(
            self.preprocessing,
            num_patches,
            mask_cfg,
            augmentations=self.augmentations,
            training=training,
        )

    @torch.inference_mode()
    def evaluate(self, dataloaders):
        pass

    def plot(self):
        pass
    
