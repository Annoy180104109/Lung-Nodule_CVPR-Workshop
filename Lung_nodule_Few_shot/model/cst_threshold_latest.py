import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.pl_module import FSCSModule
from model.module.cst import CorrelationTransformer
import model.backbone.dino.vision_transformer as vits


class ClfSegTransformer(FSCSModule):
    """
    Improvements for LIDC pseudo:
    1) Pseudo labels are NOT from support CLS token anymore.
       Instead we build a masked prototype from SUPPORT MASK region (support is still GT in few-shot episodes),
       then score QUERY pixels by cosine similarity to that prototype.

    2) Pseudo supervision uses SOFT targets (BCE on fg logit) + confidence weighting (more stable than hard CE thresholds).

    3) Prints batch tensor shapes ONCE (first forward call on rank 0).
    """

    def __init__(self, args):
        super().__init__(args)

        # ---- DINO backbone ----
        self.backbone = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.backbone.load_state_dict(state_dict, strict=True)

        self.nlayer = 12
        self.nhead = 6
        self.imgsize = args.imgsize
        self.sptsize = max(12, int(args.imgsize // 32))
        self.sup = args.sup  # "mask" or "pseudo"

        # freeze backbone
        self.backbone.eval()
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False

        # segmentation learner
        self.learner = CorrelationTransformer([self.nhead * self.nlayer], args.way)

        # query classification head
        embed_dim = getattr(self.backbone, "embed_dim", 384)
        self.cls_head = nn.Linear(embed_dim, 5)

        # pseudo masks cache (SOFT score maps): [B,way,H,W] float in [0,1]
        self._last_query_pmask_all = None

        # one-time debug flags
        self._printed_batch_shapes = False
        self._dbg_printed_loss = False

    # -------------------------
    # forward
    # -------------------------
    def forward(self, batch):
        """
        Returns:
          output_cls:  [B,5]
          output_masks:[B,way,2,H,W]
        """
        # ---- print batch shapes once (global rank 0 only) ----
        if (not self._printed_batch_shapes) and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
            self._printed_batch_shapes = True
            print("===== BATCH SHAPES (first forward) =====")
            for k, v in batch.items():
                if torch.is_tensor(v):
                    print(f"{k:22s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
                else:
                    print(f"{k:22s} type={type(v)}")
            print("=======================================")

        qry_img = batch["query_img"]
        shot = batch["support_imgs"].shape[2]

        # query classification
        with torch.no_grad():
            qry_last = self.backbone.get_intermediate_layers(qry_img, n=1)[0]  # [B,1+HW,C]
            qry_cls_feat = qry_last[:, 0, :]                                   # [B,C]
        output_cls = self.cls_head(qry_cls_feat)                                # [B,5]

        # segmentation: average across shots
        out_sum = None
        for s_idx in range(shot):
            out_masks = self._forward_one_shot(batch, shot_idx=s_idx)          # [B,way,2,H,W]
            out_sum = out_masks if out_sum is None else (out_sum + out_masks)
        output_masks = out_sum / float(shot)

        return output_cls, output_masks

    def _forward_one_shot(self, batch, shot_idx=0):
        """
        Returns:
          mask_logits: [B,way,2,H,W]
        """
        B0 = batch["query_img"].shape[0]
        way = self.args.way

        # supports: [B,way,shot,3,H,W] -> pick shot -> [B,way,3,H,W]
        support_imgs = batch["support_imgs"][:, :, shot_idx]

        # IMPORTANT: ALWAYS load support_masks (even for pseudo)
        # because we use them to build a masked prototype for pseudo labels.
        support_masks = batch["support_masks"][:, :, shot_idx]  # [B,way,H,W]

        spt_img = rearrange(support_imgs, "b n c h w -> (b n) c h w")          # [B*way,3,H,W]
        spt_mask = rearrange(support_masks, "b n h w -> (b n) h w")            # [B*way,H,W]
        qry_img = batch["query_img"]

        with torch.no_grad():
            # For pseudo we need QKV; for mask supervision we don't.
            qry_feats = self.extract_dino_feats(qry_img, return_qkv=(self.sup == "pseudo"))
            spt_feats = self.extract_dino_feats(spt_img, return_qkv=(self.sup == "pseudo"))

            # ---------- pseudo label cache (masked prototype) ----------
            if self.sup == "pseudo":
                qry_qkv, qry_feats = qry_feats
                spt_qkv, spt_feats = spt_feats

                # qry_qkv shape expected like [3, B, N, L, C]; repeat B dimension by way -> [3, B*way, N, L, C]
                qry_qkv = qry_qkv.repeat_interleave(way, dim=1)

                resize = (self.imgsize, self.imgsize) if self.training else self._get_org_hw(batch)

                # produce query score map for each (episode,way): [B*way,H,W] in [0,1]
                qry_score = self.generate_pseudo_score_from_support_mask(
                    qry_qkv=qry_qkv,
                    spt_qkv=spt_qkv,
                    spt_mask=spt_mask,     # [B*way,H,W] 0/1
                    resize=resize
                )

                # store as [B,way,H,W]
                qry_score = qry_score.view(B0, way, *qry_score.shape[-2:])
                self._last_query_pmask_all = qry_score

            # stack intermediate layers (same as your original)
            qry_feats = torch.stack(qry_feats, dim=1)                     # [B,L,1+HW,C]
            spt_feats = torch.stack(spt_feats, dim=1)                     # [B*way,L,1+HW,C]
            qry_feats = qry_feats.repeat_interleave(way, dim=0)           # [B*way,L,1+HW,C]

            Bn, L, T, C = spt_feats.shape                                 # Bn = B*way
            h = w = int(self.imgsize // 8)
            ch = int(C // self.nhead)

            qry_feat = qry_feats.reshape(Bn * L, T, C)[:, 1:, :]
            spt_feat = spt_feats.reshape(Bn * L, T, C)[:, 1:, :]
            spt_cls = spt_feats.reshape(Bn * L, T, C)[:, 0, :]

            qry_feat = rearrange(qry_feat, "b p (n c) -> b n p c", n=self.nhead, c=ch)

            spt_feat = rearrange(spt_feat, "b (h w) d -> b d h w", h=h, w=w)
            spt_feat = F.interpolate(spt_feat, (self.sptsize, self.sptsize), mode="bilinear", align_corners=True)
            spt_feat = rearrange(spt_feat, "b (n c) h w -> b n (h w) c", n=self.nhead, c=ch)

            spt_cls = rearrange(spt_cls, "b (n c) -> b n 1 c", n=self.nhead, c=ch)
            spt_feat = torch.cat([spt_cls, spt_feat], dim=2)

            qry_feat = F.normalize(qry_feat, p=2, dim=-1)
            spt_feat = F.normalize(spt_feat, p=2, dim=-1)

            headwise_corr = torch.einsum("b n q c, b n s c -> b n q s", qry_feat, spt_feat)
            headwise_corr = rearrange(headwise_corr, "(b l) n q s -> b (n l) q s", b=Bn, l=L)

        # learner expects spt_mask as supervision for support (binary/0-1 mask)
        _, mask_logits = self.learner(headwise_corr, spt_mask)            # [B*way,2,h,w]
        mask_logits = self.upsample_logit_mask(mask_logits, batch)        # [B*way,2,H,W]
        mask_logits = mask_logits.view(B0, way, *mask_logits.shape[1:])   # [B,way,2,H,W]
        return mask_logits

    # -------------------------
    # Objective
    # -------------------------
    def compute_objective(self, output_cls, output_masks, gt_cls, gt_mask, gt_way, lambda_cls=1.0):
        """
        output_cls:   [B,5]
        output_masks: [B,way,2,H,W]
        gt_cls:       [B] 0..4
        gt_mask:      [B,H,W]
        gt_way:       [B] 0..way-1
        """
        cls_loss = F.cross_entropy(output_cls, gt_cls.long())

        B, way, C, H, W = output_masks.shape
        idx = gt_way.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
        seg_logits = output_masks.gather(1, idx).squeeze(1)  # [B,2,H,W]

        if self.sup == "pseudo":
            if self._last_query_pmask_all is None:
                raise RuntimeError("sup='pseudo' but _last_query_pmask_all is None (pseudo score not generated).")

            # p in [0,1] for the GT way
            idx2 = gt_way.view(B, 1, 1, 1).expand(-1, 1, H, W)
            p = self._last_query_pmask_all.gather(1, idx2).squeeze(1).clamp(1e-4, 1 - 1e-4)  # [B,H,W]

            # soft pseudo loss on FG logit with confidence weighting
            fg_logit = seg_logits[:, 1]  # [B,H,W]
            conf = (p - 0.5).abs() * 2.0  # [0,1], uncertain pixels contribute less
            seg_loss = (F.binary_cross_entropy_with_logits(fg_logit, p, reduction="none") * conf).mean()

            # optional debug once
            if (not self._dbg_printed_loss) and self.training and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
                self._dbg_printed_loss = True
                with torch.no_grad():
                    print("[PSEUDO-LOSS] p(min/mean/max)=(%.3f/%.3f/%.3f) conf(mean)=%.3f"
                          % (p.min().item(), p.mean().item(), p.max().item(), conf.mean().item()))
        else:
            seg_target = gt_mask.long()
            seg_loss = F.cross_entropy(seg_logits, seg_target)

        return seg_loss + lambda_cls * cls_loss

    # -------------------------
    # predict_cls_seg for test
    # -------------------------
    def predict_cls_seg(self, batch, nshot):
        with torch.no_grad():
            qry_img = batch["query_img"]
            qry_last = self.backbone.get_intermediate_layers(qry_img, n=1)[0]
            qry_cls_feat = qry_last[:, 0, :]
            output_cls = self.cls_head(qry_cls_feat)
            pred_cls = torch.argmax(output_cls, dim=1)

        shot = batch["support_imgs"].shape[2]
        nshot = min(nshot, shot)

        out_sum = None
        for s_idx in range(nshot):
            out_masks = self._forward_one_shot(batch, shot_idx=s_idx)
            out_sum = out_masks if out_sum is None else (out_sum + out_masks)
        out_masks = out_sum / float(nshot)  # [B,way,2,H,W]

        support_classes = batch["support_classes"].long()  # [B,way] 1..5
        pred_way = (support_classes == (pred_cls + 1).view(-1, 1)).long().argmax(dim=1)
        pred_seg = self._predict_mask_from_way(out_masks, pred_way)

        return pred_cls, pred_seg

    def _predict_mask_from_way(self, output_masks, way_idx):
        with torch.no_grad():
            B, way, C, H, W = output_masks.shape
            idx = way_idx.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
            seg_logits = output_masks.gather(1, idx).squeeze(1)  # [B,2,H,W]
            return seg_logits.argmax(dim=1).long()

    # -------------------------
    # DINO feature extraction
    # -------------------------
    def extract_dino_feats(self, img, return_qkv=False):
        return self.backbone.get_intermediate_layers(img, n=self.nlayer, return_qkv=return_qkv)

    # -------------------------
    # Pseudo score from masked support prototype
    # -------------------------
    def generate_pseudo_score_from_support_mask(self, qry_qkv, spt_qkv, spt_mask, resize=(96, 96)):
        """
        qry_qkv:  [3, B*way, N, L, C]
        spt_qkv:  [3, B*way, N, L, C]
        spt_mask: [B*way, H_img, W_img] (0/1)
        returns:
          qry_score: [B*way, H_img, W_img] float in [0,1]
        """
        # use KEYS (index 1). remove CLS token -> patches only
        qry_key = qry_qkv[1, :, :, 1:, :]  # [B*way, N, HW, C]
        spt_key = spt_qkv[1, :, :, 1:, :]  # [B*way, N, HW, C]

        h = w = int(self.imgsize // 8)  # patch grid

        qry_key = rearrange(qry_key, "b n (h w) c -> b n h w c", h=h, w=w)
        spt_key = rearrange(spt_key, "b n (h w) c -> b n h w c", h=h, w=w)

        qry_key = F.normalize(qry_key, p=2, dim=-1)
        spt_key = F.normalize(spt_key, p=2, dim=-1)

        # downsample support mask to patch grid (nearest!)
        m = spt_mask.float().unsqueeze(1)  # [B*way,1,H,W]
        m = F.interpolate(m, size=(h, w), mode="nearest").squeeze(1)  # [B*way,h,w]

        # avoid empty mask
        denom = m.sum(dim=(1, 2)).clamp(min=1.0)  # [B*way]

        # masked prototype per head: mean support key inside mask
        m5 = m[:, None, :, :, None]  # [B*way,1,h,w,1]
        proto = (spt_key * m5).sum(dim=(2, 3)) / denom[:, None, None]  # [B*way,N,C]
        proto = F.normalize(proto, p=2, dim=-1)

        # similarity map
        sim = torch.einsum("b n h w c, b n c -> b n h w", qry_key, proto)  # [B*way,N,h,w]
        sim = sim.mean(dim=1, keepdim=True)  # [B*way,1,h,w]

        # upsample to image size and map [-1,1]->[0,1]
        sim = F.interpolate(sim, size=resize, mode="bilinear", align_corners=True).squeeze(1)  # [B*way,H,W]
        qry_score = ((sim + 1.0) * 0.5).clamp(0.0, 1.0)

        return qry_score

    # -------------------------
    # Upsampling helpers
    # -------------------------
    def _get_org_hw(self, batch):
        org = batch.get("org_query_imsize", None)
        if org is None:
            return tuple(batch["query_img"].shape[-2:])

        if torch.is_tensor(org):
            if org.numel() == 2:
                a = int(org.view(-1)[0].item())
                b = int(org.view(-1)[1].item())
                W, H = a, b
                return (H, W)

            if org.ndim == 2 and org.shape[-1] == 2:
                a = int(org[0, 0].item())
                b = int(org[0, 1].item())
                W, H = a, b
                return (H, W)

            if org.ndim == 2 and org.shape[0] == 2:
                a = int(org[0, 0].item())
                b = int(org[1, 0].item())
                W, H = a, b
                return (H, W)

            return tuple(batch["query_img"].shape[-2:])

        try:
            if isinstance(org, (list, tuple)) and len(org) >= 2:
                W = int(org[0])
                H = int(org[1])
                return (H, W)
        except Exception:
            pass

        return tuple(batch["query_img"].shape[-2:])

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch["query_img"].shape[-2:]
        else:
            spatial_size = self._get_org_hw(batch)
        return F.interpolate(logit_mask, spatial_size, mode="bilinear", align_corners=True)

    # -------------------------
    # misc
    # -------------------------
    def train_mode(self):
        self.train()
        self.backbone.eval()

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.args.lr)