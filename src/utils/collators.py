from .masks import jepa_mask
from torch.utils.data import default_collate


class IdentityCollator:

    def __init__(self, training=True):
        self.training = training

    def __call__(batch):
        return batch


class SupervisedCollator:

    def __init__(self, preprocessing, augmentations, training=True):
        self.preprocessing = preprocessing
        self.augmentations = augmentations
        self.training = training

    def __call__(self, batch):

        for transform in self.preprocessing["y"]:
            batch.labels = transform.forward(batch.labels)

        if batch.images is not None:
            for transform in self.preprocessing["x"]:
                batch.images = transform.forward(batch.images)

            if self.training:
                for aug in self.augmentations:
                    batch.images = aug(batch.images)

        if (batch.summaries is not None) and self.augmentations:
            # select a random augmentation
            idx = torch.randint(batch.summaries.size(1), ())
            batch.summaries = batch.summaries[:, idx]

        return batch


class SummarizationCollator:

    def __init__(self, preprocessing, training=True):
        self.preprocessing = preprocessing
        self.training = training

    def __call__(self, images):

        images = default_collate(images)
        # preprocess
        for transform in self.preprocessing["x"]:
            images = transform.forward(images)

        return images


class JEPACollator:

    # CHANGED 2026-05-13: added `augmentations` parameter so that JEPA pretraining
    # actually applies the augmentations listed in cfg.training.augmentations.
    # Previously augmentations were loaded into self.augmentations by the experiment
    # but never forwarded here and never applied.
    def __init__(self, preprocessing, num_patches, mask_cfg, augmentations=None, training=True):
        self.preprocessing = preprocessing
        self.num_patches = num_patches
        self.mask_cfg = mask_cfg
        self.augmentations = augmentations if augmentations is not None else []
        self.training = training

    def __call__(self, batch):

        # preprocess images
        for transform in self.preprocessing["x"]:
            batch.images = transform.forward(batch.images)

        # CHANGED 2026-05-13: apply augmentations on the training split only,
        # after preprocessing and before mask sampling. Mirrors SupervisedCollator.
        # Teacher and student see the same augmented image because JEPA.batch_loss
        # passes one `images` tensor to both encoders.
        if self.training:
            for aug in self.augmentations:
                batch.images = aug(batch.images)

        # sample masks
        tgt_masks, ctx_masks = jepa_mask(
            self.num_patches,
            self.mask_cfg,
            batch_size=len(batch),
            device=batch.device,
        )

        return batch.images, tgt_masks, ctx_masks
