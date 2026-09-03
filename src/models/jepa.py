import copy
import torch
import torch.nn.functional as F

from src.models.base_model import Model
# CHANGED 2026-05-13: dropped `augmentations` import — augmentations now live in
# JEPACollator (src/utils/collators.py), not in the model.
from src.utils import masks


class JEPA(Model):

    def __init__(
        self,
        net,
        predictor,
        sim="l1",
        ema_momentum=0.9997,
        momentum_schedule=True,
        init_tgt_as_ctx=True,
        summary_net=None,
        # CHANGED 2026-05-18: new flag to gate the EMA-update fix.
        # The original code `pt = tau * pt + (1 - tau) * ps` is a Python no-op
        # (rebinds the local variable instead of mutating the parameter), so
        # the target encoder stays frozen at initialisation throughout training.
        # Default True applies the corrected in-place update; set False to
        # exactly reproduce pre-2026-05-18 ("frozen random teacher") runs.
        use_correct_ema_update=True,
    ):
        super().__init__(net=net, summary_net=summary_net)
        self.predictor = predictor
        self.ema_momentum = ema_momentum
        self.momentum_schedule = momentum_schedule
        self.use_correct_ema_update = use_correct_ema_update
        self.ctx_encoder = self.net
        self.tgt_encoder = copy.deepcopy(self.ctx_encoder)
        if not init_tgt_as_ctx:
            # re-initialize weights
            for module in self.tgt_encoder.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

        # CHANGED 2026-05-13: removed dead `self.augment = augmentations.RotateAndReflect()`.
        # It was constructed but never called in batch_loss; only one of the two
        # configured transforms and without include_identity=True. Augmentations
        # now live in JEPACollator.

        match sim:
            case "l2":
                self.sim = lambda x1, x2: -F.mse_loss(x1, x2)
            case "l1":
                self.sim = lambda x1, x2: -F.l1_loss(x1, x2)
            case "smooth_l1":
                self.sim = lambda x1, x2: -F.smooth_l1_loss(x1, x2)

    def batch_loss(self, batch):

        images, tgt_masks, ctx_masks = batch

        # get target token embeddings
        with torch.no_grad():
            tgt_tokens = self.tgt_encoder(images)

        loss = 0.0
        for ctx_mask, tgt_mask in zip(ctx_masks, tgt_masks):
            # WARNING: Assumes each target mask has it's own context. Repeat ctx_mask otherwise

            # get context token embeddings and predict
            ctx_tokens = self.ctx_encoder(images, mask=ctx_mask)
            prd_tokens = self.predictor(ctx_tokens, ctx_mask, tgt_mask)

            # keep only target tokens in current block
            local_tgt_tokens = masks.gather_tokens(tgt_tokens, tgt_mask)

            # similarity loss
            loss += -self.sim(prd_tokens, local_tgt_tokens)

        return loss

    def update(
        self, loss, optimizer, scaler, step=None, total_steps=None, gradient_norm=None
    ):

        # student update
        super().update(loss, optimizer, scaler, gradient_norm=gradient_norm)

        # teacher update via exponential moving average of student
        tau = self.ema_momentum
        if self.momentum_schedule:  # linear increase to tau=1
            frac = step / total_steps
            tau = tau + (1 - tau) * frac

        # CHANGED 2026-05-18: fixed in-place EMA update.
        # The original line below silently no-ops because `pt = tau * pt + ...`
        # rebinds the local variable rather than mutating the nn.Parameter
        # stored in self.tgt_encoder. The in-place version writes through to
        # the parameter tensor itself.
        if self.use_correct_ema_update:
            with torch.no_grad():
                for ps, pt in zip(
                    self.ctx_encoder.parameters(),
                    self.tgt_encoder.parameters(),
                ):
                    pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)
        else:
            # Legacy no-op path — preserved for exact reproduction of runs
            # before 2026-05-18. The target encoder remains frozen at
            # initialisation (equivalent to ema_momentum=1.0).
            for ps, pt in zip(
                self.ctx_encoder.parameters(),
                self.tgt_encoder.parameters(),
            ):
                pt = tau * pt + (1 - tau) * ps

    def forward(self, x, mask=False):
        return self.ctx_encoder(x, mask=mask)

    @torch.inference_mode()
    def embed(self, x):
        return self.ctx_encoder(x)
