# # =========================
# # cst.py  (ClfSegTransformer)  FULL FILE
# # Supports vit-small (DINO) OR resnet50/resnet101 (torchvision)
# # =========================
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange

# from model.pl_module import FSCSModule
# from model.module.cst import CorrelationTransformer

# # ViT/DINO backbone (existing)
# import model.backbone.dino.vision_transformer as vits

# # ResNet backbones (torchvision)
# import torchvision.models as tvm


# class ResNet50Backbone(nn.Module):
#     """
#     Returns:
#       feat8: layer2 output (stride=8),  shape [B, 512, H/8,  W/8]
#       cls:   pooled layer4 (stride=32), shape [B, 2048]
#     """
#     def __init__(self, pretrained=True):
#         super().__init__()
#         weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
#         m = tvm.resnet50(weights=weights)

#         self.conv1 = m.conv1
#         self.bn1 = m.bn1
#         self.relu = m.relu
#         self.maxpool = m.maxpool
#         self.layer1 = m.layer1  # stride 4
#         self.layer2 = m.layer2  # stride 8
#         self.layer3 = m.layer3  # stride 16
#         self.layer4 = m.layer4  # stride 32

#         self.embed_dim = 2048

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         feat8 = self.layer2(x)                 # [B,512,H/8,W/8]
#         x = self.layer3(feat8)
#         x = self.layer4(x)                     # [B,2048,H/32,W/32]
#         cls = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B,2048]
#         return {"feat8": feat8, "cls": cls}


# class ResNet101Backbone(nn.Module):
#     """
#     Returns:
#       feat8: layer2 output (stride=8),  shape [B, 512, H/8,  W/8]
#       cls:   pooled layer4 (stride=32), shape [B, 2048]
#     """
#     def __init__(self, pretrained=True):
#         super().__init__()
#         weights = tvm.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
#         m = tvm.resnet101(weights=weights)

#         self.conv1 = m.conv1
#         self.bn1 = m.bn1
#         self.relu = m.relu
#         self.maxpool = m.maxpool
#         self.layer1 = m.layer1  # stride 4
#         self.layer2 = m.layer2  # stride 8
#         self.layer3 = m.layer3  # stride 16
#         self.layer4 = m.layer4  # stride 32

#         self.embed_dim = 2048

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         feat8 = self.layer2(x)                 # [B,512,H/8,W/8]
#         x = self.layer3(feat8)
#         x = self.layer4(x)                     # [B,2048,H/32,W/32]
#         cls = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B,2048]
#         return {"feat8": feat8, "cls": cls}


# class ClfSegTransformer(FSCSModule):
#     """
#     Supports:
#       - sup="mask"   : query supervision uses GT pixel mask (cross-entropy on 2-class logits)
#       - sup="pseudo" : query supervision uses pseudo prob map generated from support GT mask region

#     NEW:
#       - args.pseudo_ratio in [0,1] controls PIXEL-LEVEL mixed supervision when sup="pseudo":
#           pseudo_ratio = 0.0  -> 100% GT pixels
#           pseudo_ratio = 1.0  -> 100% pseudo pixels
#           between      -> per-pixel Bernoulli mixing of pseudo vs GT target

#     Notes:
#       - For the mixing to truly use GT, pl_module.py must pass real query GT mask as gt_mask.
#     """

#     def __init__(self, args):
#         super().__init__(args)

#         self.imgsize = args.imgsize
#         self.sup = args.sup  # "mask" or "pseudo"

#         # pseudo controls (safe defaults)
#         self.pseudo_tau = float(getattr(args, "pseudo_tau", 10.0))   # sigmoid temperature
#         self.pseudo_head_pool = str(getattr(args, "pseudo_head_pool", "max"))  # "mean" or "max"
#         self.pseudo_thr_mode = str(getattr(args, "pseudo_thr_mode", "quantile"))  # "quantile" or "meanstd"
#         self.pseudo_thr_q = float(getattr(args, "pseudo_thr_q", 0.85))  # quantile for adaptive threshold
#         self.pseudo_thr_k = float(getattr(args, "pseudo_thr_k", 0.5))    # mean+ k*std

#         # cache: pseudo prob maps [B,way,H,W] in [0,1]
#         self._last_query_pmask_all = None

#         # debug flags
#         self._printed_batch_shapes = False
#         self._dbg_printed_qkv = False
#         self._dbg_printed_pseudo_stats = False

#         # -------------------------
#         # Backbone selection
#         # -------------------------
#         bb = str(getattr(args, "backbone", "vit-small")).lower()
#         self.backbone_name = bb

#         if bb in ["vit-small", "vit_small", "vit"]:
#             self.backbone_type = "vit"

#             # ---- DINO backbone (as you had) ----
#             self.backbone = vits.__dict__["vit_small"](patch_size=8, num_classes=0)
#             url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
#             state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
#             self.backbone.load_state_dict(state_dict, strict=True)

#             self.nlayer = 12
#             self.nhead = 6
#             self.patch_stride = 8
#             embed_dim = getattr(self.backbone, "embed_dim", 384)

#         elif bb in ["resnet50", "resnet-50", "r50"]:
#             self.backbone_type = "resnet"

#             self.backbone = ResNet50Backbone(pretrained=True)

#             # for correlation we use ONE feature level (feat8) -> keep learner channel dim = 1
#             self.nlayer = 1
#             self.nhead = 1
#             self.patch_stride = 8
#             embed_dim = 2048

#         elif bb in ["resnet101", "resnet-101", "r101"]:
#             self.backbone_type = "resnet"

#             self.backbone = ResNet101Backbone(pretrained=True)

#             # for correlation we use ONE feature level (feat8) -> keep learner channel dim = 1
#             self.nlayer = 1
#             self.nhead = 1
#             self.patch_stride = 8
#             embed_dim = 2048

#         else:
#             raise ValueError(
#                 f"Unsupported backbone: {args.backbone}. Use 'vit-small', 'resnet50', or 'resnet101'."
#             )

#         # freeze backbone
#         self.backbone.eval()
#         for p in self.backbone.parameters():
#             p.requires_grad = False

#         # segmentation learner
#         self.sptsize = max(12, int(args.imgsize // 32))
#         self.learner = CorrelationTransformer([self.nhead * self.nlayer], args.way)

#         # query classification head
#         self.cls_head = nn.Linear(embed_dim, 5)

#     # -------------------------
#     # forward
#     # -------------------------
#     def forward(self, batch):
#         """
#         Returns:
#           output_cls:  [B,5]
#           output_masks:[B,way,2,H,W]
#         """
#         # ---- print batch shapes once (rank 0 only) ----
#         if (not self._printed_batch_shapes) and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
#             self._printed_batch_shapes = True
#             print("===== BATCH SHAPES (first forward) =====")
#             for k, v in batch.items():
#                 if torch.is_tensor(v):
#                     print(f"{k:22s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
#                 else:
#                     print(f"{k:22s} type={type(v)}")
#             print("=======================================")

#         qry_img = batch["query_img"]
#         shot = batch["support_imgs"].shape[2]

#         # ---- query classification ----
#         with torch.no_grad():
#             if self.backbone_type == "vit":
#                 qry_last = self.backbone.get_intermediate_layers(qry_img, n=1)[0]  # [B,1+HW,C]
#                 qry_cls_feat = qry_last[:, 0, :]                                   # [B,C]
#             else:
#                 out = self.backbone(qry_img)
#                 qry_cls_feat = out["cls"]                                          # [B,2048]

#         output_cls = self.cls_head(qry_cls_feat)                                   # [B,5]

#         # ---- segmentation: average across shots ----
#         out_sum = None
#         for s_idx in range(shot):
#             out_masks = self._forward_one_shot(batch, shot_idx=s_idx)             # [B,way,2,H,W]
#             out_sum = out_masks if out_sum is None else (out_sum + out_masks)
#         output_masks = out_sum / float(shot)

#         return output_cls, output_masks

#     def _forward_one_shot(self, batch, shot_idx=0):
#         """
#         Returns:
#           mask_logits: [B,way,2,H,W]
#         """
#         B0 = batch["query_img"].shape[0]
#         way = self.args.way

#         # supports: [B,way,shot,3,H,W] -> pick shot -> [B,way,3,H,W]
#         support_imgs = batch["support_imgs"][:, :, shot_idx]
#         support_masks = batch["support_masks"][:, :, shot_idx]  # [B,way,H,W]

#         spt_img = rearrange(support_imgs, "b n c h w -> (b n) c h w")   # [B*way,3,H,W]
#         spt_mask = rearrange(support_masks, "b n h w -> (b n) h w")     # [B*way,H,W]
#         qry_img = batch["query_img"]

#         if self.backbone_type == "vit":
#             return self._forward_one_shot_vit(batch, qry_img, spt_img, spt_mask, B0, way)
#         else:
#             return self._forward_one_shot_resnet(batch, qry_img, spt_img, spt_mask, B0, way)

#     # -------------------------
#     # ViT path
#     # -------------------------
#     def _forward_one_shot_vit(self, batch, qry_img, spt_img, spt_mask, B0, way):
#         with torch.no_grad():
#             qry_feats = self.extract_vit_feats(qry_img, return_qkv=(self.sup == "pseudo"))
#             spt_feats = self.extract_vit_feats(spt_img, return_qkv=(self.sup == "pseudo"))

#             # ---------- pseudo label cache (masked prototype) ----------
#             if self.sup == "pseudo":
#                 qry_qkv, qry_layers = qry_feats
#                 spt_qkv, spt_layers = spt_feats

#                 # Debug qkv once
#                 if (not self._dbg_printed_qkv) and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
#                     self._dbg_printed_qkv = True
#                     print("[QKV DEBUG] qry_qkv type:", type(qry_qkv))
#                     if torch.is_tensor(qry_qkv):
#                         print("[QKV DEBUG] qry_qkv shape:", tuple(qry_qkv.shape))
#                     else:
#                         try:
#                             print("[QKV DEBUG] len(qry_qkv):", len(qry_qkv))
#                             print("[QKV DEBUG] qry_qkv[-1] shape:", tuple(qry_qkv[-1].shape))
#                         except Exception as e:
#                             print("[QKV DEBUG] cannot inspect list qkv:", repr(e))

#                 qry_qkv = self._select_last_layer_qkv(qry_qkv)
#                 spt_qkv = self._select_last_layer_qkv(spt_qkv)

#                 # make query qkv have B*way in batch dim
#                 if qry_qkv.dim() != 5:
#                     raise RuntimeError(
#                         f"Unexpected qry_qkv dim={qry_qkv.dim()} (expected 5). Got shape={tuple(qry_qkv.shape)}"
#                     )
#                 qry_qkv = qry_qkv.repeat_interleave(way, dim=1)  # dim=1 is batch

#                 resize = (self.imgsize, self.imgsize) if self.training else self._get_org_hw(batch)

#                 qry_score = self.generate_pseudo_score_from_support_mask_vit(
#                     qry_qkv=qry_qkv,
#                     spt_qkv=spt_qkv,
#                     spt_mask=spt_mask,
#                     resize=resize
#                 )

#                 qry_score = qry_score.view(B0, way, *qry_score.shape[-2:])  # [B,way,H,W]
#                 self._last_query_pmask_all = qry_score

#                 # continue with layer stacks for correlation learner
#                 qry_feats = qry_layers
#                 spt_feats = spt_layers

#             # stack intermediate layers
#             qry_feats = torch.stack(qry_feats, dim=1)                 # [B,L,1+HW,C]
#             spt_feats = torch.stack(spt_feats, dim=1)                 # [B*way,L,1+HW,C]
#             qry_feats = qry_feats.repeat_interleave(way, dim=0)       # [B*way,L,1+HW,C]

#             Bn, L, T, C = spt_feats.shape                             # Bn = B*way
#             h = w = int(self.imgsize // self.patch_stride)
#             ch = int(C // self.nhead)

#             qry_feat = qry_feats.reshape(Bn * L, T, C)[:, 1:, :]
#             spt_feat = spt_feats.reshape(Bn * L, T, C)[:, 1:, :]
#             spt_cls = spt_feats.reshape(Bn * L, T, C)[:, 0, :]

#             qry_feat = rearrange(qry_feat, "b p (n c) -> b n p c", n=self.nhead, c=ch)

#             spt_feat = rearrange(spt_feat, "b (h w) d -> b d h w", h=h, w=w)
#             spt_feat = F.interpolate(spt_feat, (self.sptsize, self.sptsize), mode="bilinear", align_corners=True)
#             spt_feat = rearrange(spt_feat, "b (n c) h w -> b n (h w) c", n=self.nhead, c=ch)

#             spt_cls = rearrange(spt_cls, "b (n c) -> b n 1 c", n=self.nhead, c=ch)
#             spt_feat = torch.cat([spt_cls, spt_feat], dim=2)

#             qry_feat = F.normalize(qry_feat, p=2, dim=-1)
#             spt_feat = F.normalize(spt_feat, p=2, dim=-1)

#             headwise_corr = torch.einsum("b n q c, b n s c -> b n q s", qry_feat, spt_feat)
#             headwise_corr = rearrange(headwise_corr, "(b l) n q s -> b (n l) q s", b=Bn, l=L)

#         # learner expects spt_mask supervision for support
#         _, mask_logits = self.learner(headwise_corr, spt_mask)            # [B*way,2,h,w]
#         mask_logits = self.upsample_logit_mask(mask_logits, batch)        # [B*way,2,H,W]
#         mask_logits = mask_logits.view(B0, way, *mask_logits.shape[1:])   # [B,way,2,H,W]
#         return mask_logits

#     # -------------------------
#     # ResNet path (shared for ResNet50/101)
#     # -------------------------
#     def _forward_one_shot_resnet(self, batch, qry_img, spt_img, spt_mask, B0, way):
#         with torch.no_grad():
#             qry_out = self.backbone(qry_img)
#             spt_out = self.backbone(spt_img)

#             qry_feat8 = qry_out["feat8"]      # [B,512,h,w]
#             spt_feat8 = spt_out["feat8"]      # [B*way,512,h,w]

#             # pseudo cache
#             if self.sup == "pseudo":
#                 qry_feat8_rep = qry_feat8.repeat_interleave(way, dim=0)  # [B*way,512,h,w]
#                 resize = (self.imgsize, self.imgsize) if self.training else self._get_org_hw(batch)

#                 qry_score = self.generate_pseudo_score_from_support_mask_resnet(
#                     qry_feat8=qry_feat8_rep,
#                     spt_feat8=spt_feat8,
#                     spt_mask=spt_mask,
#                     resize=resize,
#                 )
#                 qry_score = qry_score.view(B0, way, *qry_score.shape[-2:])  # [B,way,H,W]
#                 self._last_query_pmask_all = qry_score

#             # correlation (single "channel")
#             qry_feat8 = qry_feat8.repeat_interleave(way, dim=0)   # [B*way,512,h,w]

#             qry_feat8 = F.normalize(qry_feat8, p=2, dim=1)
#             spt_feat8 = F.normalize(spt_feat8, p=2, dim=1)

#             Bn, Cc, h, w = spt_feat8.shape
#             q = qry_feat8.flatten(2)                     # [Bn,C,hw]
#             s = spt_feat8.flatten(2)                     # [Bn,C,hw]

#             # "cls-like" support token
#             s_cls = spt_feat8.mean(dim=(2, 3), keepdim=False).unsqueeze(-1)  # [Bn,C,1]
#             s = torch.cat([s_cls, s], dim=-1)                                # [Bn,C,1+hw]

#             corr = torch.einsum("b c q, b c s -> b q s", q, s)               # [Bn,hw,1+hw]
#             headwise_corr = corr.unsqueeze(1)                                 # [Bn,1,hw,1+hw]

#         _, mask_logits = self.learner(headwise_corr, spt_mask)            # [B*way,2,h',w'] (learner-defined)
#         mask_logits = self.upsample_logit_mask(mask_logits, batch)        # [B*way,2,H,W]
#         mask_logits = mask_logits.view(B0, way, *mask_logits.shape[1:])   # [B,way,2,H,W]
#         return mask_logits

#     # -------------------------
#     # Objective
#     # -------------------------
#     def compute_objective(self, output_cls, output_masks, gt_cls, gt_mask, gt_way, lambda_cls=1.0):
#         cls_loss = F.cross_entropy(output_cls, gt_cls.long())

#         B, way, C, H, W = output_masks.shape
#         idx = gt_way.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
#         seg_logits = output_masks.gather(1, idx).squeeze(1)  # [B,2,H,W]

#         if self.sup == "pseudo":
#             if self._last_query_pmask_all is None:
#                 raise RuntimeError("sup='pseudo' but _last_query_pmask_all is None (pseudo score not generated).")

#             idx2 = gt_way.view(B, 1, 1, 1).expand(-1, 1, H, W)
#             p = self._last_query_pmask_all.gather(1, idx2).squeeze(1).clamp(1e-4, 1 - 1e-4)  # [B,H,W]

#             y = gt_mask.float().clamp(0.0, 1.0)  # [B,H,W]

#             r = float(getattr(self.args, "pseudo_ratio", 1.0))
#             r = max(0.0, min(1.0, r))

#             if r == 0.0:
#                 target = y
#             elif r == 1.0:
#                 target = p
#             else:
#                 m = (torch.rand_like(p) < r).float()
#                 target = m * p + (1.0 - m) * y  # [B,H,W] soft

#             fg_logit = seg_logits[:, 1]  # [B,H,W]
#             conf = (target - 0.5).abs() * 2.0  # [0,1]
#             seg_loss = (F.binary_cross_entropy_with_logits(fg_logit, target, reduction="none") * conf).mean()

#             if (not self._dbg_printed_pseudo_stats) and self.training and (not hasattr(self, "trainer") or self.trainer.is_global_zero):
#                 self._dbg_printed_pseudo_stats = True
#                 with torch.no_grad():
#                     thr = self.adaptive_threshold(p)
#                     print("[MIXED-PSEUDO] pseudo_ratio={:.3f} p(min/mean/max)=({:.3f}/{:.3f}/{:.3f}) thr(mean)={:.3f}".format(
#                         r, p.min().item(), p.mean().item(), p.max().item(), thr.mean().item()
#                     ))

#         else:
#             seg_target = gt_mask.long()
#             seg_loss = F.cross_entropy(seg_logits, seg_target)

#         return seg_loss + lambda_cls * cls_loss

#     # -------------------------
#     # predict_cls_seg for test (required by FSCSModule)
#     # -------------------------
#     def predict_cls_seg(self, batch, nshot):
#         with torch.no_grad():
#             qry_img = batch["query_img"]
#             if self.backbone_type == "vit":
#                 qry_last = self.backbone.get_intermediate_layers(qry_img, n=1)[0]
#                 qry_cls_feat = qry_last[:, 0, :]
#             else:
#                 qry_cls_feat = self.backbone(qry_img)["cls"]

#             output_cls = self.cls_head(qry_cls_feat)
#             pred_cls = torch.argmax(output_cls, dim=1)

#         shot = batch["support_imgs"].shape[2]
#         nshot = min(int(nshot), int(shot))

#         out_sum = None
#         for s_idx in range(nshot):
#             out_masks = self._forward_one_shot(batch, shot_idx=s_idx)
#             out_sum = out_masks if out_sum is None else (out_sum + out_masks)
#         out_masks = out_sum / float(nshot)  # [B,way,2,H,W]

#         support_classes = batch["support_classes"].long()  # [B,way] 1..5
#         pred_way = (support_classes == (pred_cls + 1).view(-1, 1)).long().argmax(dim=1)
#         pred_seg = self._predict_mask_from_way(out_masks, pred_way)

#         return pred_cls, pred_seg

#     def _predict_mask_from_way(self, output_masks, way_idx):
#         with torch.no_grad():
#             B, way, C, H, W = output_masks.shape
#             idx = way_idx.view(B, 1, 1, 1, 1).expand(-1, 1, C, H, W)
#             seg_logits = output_masks.gather(1, idx).squeeze(1)  # [B,2,H,W]
#             return seg_logits.argmax(dim=1).long()

#     # -------------------------
#     # ViT feature extraction
#     # -------------------------
#     def extract_vit_feats(self, img, return_qkv=False):
#         return self.backbone.get_intermediate_layers(img, n=self.nlayer, return_qkv=return_qkv)

#     # -------------------------
#     # QKV selection helper (robust)
#     # -------------------------
#     @staticmethod
#     def _select_last_layer_qkv(qkv):
#         if torch.is_tensor(qkv):
#             return qkv
#         if isinstance(qkv, (list, tuple)):
#             return qkv[-1]
#         raise TypeError(f"Unsupported qkv type: {type(qkv)}")

#     # -------------------------
#     # Adaptive threshold (per-image)
#     # -------------------------
#     def adaptive_threshold(self, p_bhw: torch.Tensor) -> torch.Tensor:
#         B = p_bhw.shape[0]
#         x = p_bhw.view(B, -1)

#         if self.pseudo_thr_mode == "meanstd":
#             thr = x.mean(dim=1) + self.pseudo_thr_k * x.std(dim=1)
#         else:
#             thr = torch.quantile(x, q=self.pseudo_thr_q, dim=1)

#         return thr.clamp(0.05, 0.95)

#     def binarize_pseudo(self, p_bhw: torch.Tensor) -> torch.Tensor:
#         thr = self.adaptive_threshold(p_bhw).view(-1, 1, 1)
#         return (p_bhw >= thr).long()

#     # -------------------------
#     # Pseudo score (ViT: from QKV keys)
#     # -------------------------
#     def generate_pseudo_score_from_support_mask_vit(self, qry_qkv, spt_qkv, spt_mask, resize=(96, 96)):
#         """
#         Expected qkv tensor format:
#           [3, B*, heads, tokens, dim_head]
#         tokens = 1 + HW_patches
#         """
#         qry_key = qry_qkv[1]  # [B*way, heads, tokens, dim_head]
#         spt_key = spt_qkv[1]

#         qry_key = qry_key[:, :, 1:, :]
#         spt_key = spt_key[:, :, 1:, :]

#         h = w = int(self.imgsize // self.patch_stride)

#         qry_key = rearrange(qry_key, "b n (h w) c -> b n h w c", h=h, w=w)
#         spt_key = rearrange(spt_key, "b n (h w) c -> b n h w c", h=h, w=w)

#         qry_key = F.normalize(qry_key, p=2, dim=-1)
#         spt_key = F.normalize(spt_key, p=2, dim=-1)

#         m = spt_mask.float().unsqueeze(1)  # [B*way,1,H,W]
#         m = F.interpolate(m, size=(h, w), mode="area").squeeze(1)
#         m = (m > 0.3).float()

#         denom = m.sum(dim=(1, 2)).clamp(min=1.0)

#         m5 = m[:, None, :, :, None]
#         proto = (spt_key * m5).sum(dim=(2, 3)) / denom[:, None, None]
#         proto = F.normalize(proto, p=2, dim=-1)

#         sim = torch.einsum("b n h w c, b n c -> b n h w", qry_key, proto)

#         if self.pseudo_head_pool == "mean":
#             sim = sim.mean(dim=1, keepdim=True)
#         else:
#             sim = sim.max(dim=1, keepdim=True).values

#         sim = F.interpolate(sim, size=resize, mode="bilinear", align_corners=True).squeeze(1)
#         qry_score = torch.sigmoid(self.pseudo_tau * sim).clamp(0.0, 1.0)
#         return qry_score

#     # -------------------------
#     # Pseudo score (ResNet: from feature maps)
#     # -------------------------
#     def generate_pseudo_score_from_support_mask_resnet(self, qry_feat8, spt_feat8, spt_mask, resize=(96, 96)):
#         """
#         qry_feat8: [B*way, C, h, w]
#         spt_feat8: [B*way, C, h, w]
#         spt_mask : [B*way, H_img, W_img]
#         returns  : [B*way, H_img, W_img] in [0,1]
#         """
#         qry = F.normalize(qry_feat8, p=2, dim=1)
#         spt = F.normalize(spt_feat8, p=2, dim=1)

#         Bn, Cc, h, w = spt.shape

#         m = spt_mask.float().unsqueeze(1)                 # [Bn,1,H,W]
#         m = F.interpolate(m, size=(h, w), mode="area")    # [Bn,1,h,w]
#         m = (m > 0.3).float()

#         denom = m.sum(dim=(2, 3)).clamp(min=1.0)          # [Bn,1]

#         proto = (spt * m).sum(dim=(2, 3)) / denom         # [Bn,C]
#         proto = F.normalize(proto, p=2, dim=1)

#         sim = (qry * proto[:, :, None, None]).sum(dim=1)  # [Bn,h,w]

#         sim = sim.unsqueeze(1)
#         sim = F.interpolate(sim, size=resize, mode="bilinear", align_corners=True).squeeze(1)
#         p = torch.sigmoid(self.pseudo_tau * sim).clamp(0.0, 1.0)
#         return p

#     # -------------------------
#     # Upsampling helpers
#     # -------------------------
#     def _get_org_hw(self, batch):
#         org = batch.get("org_query_imsize", None)
#         if org is None:
#             return tuple(batch["query_img"].shape[-2:])

#         if torch.is_tensor(org):
#             if org.numel() == 2:
#                 a = int(org.view(-1)[0].item())
#                 b = int(org.view(-1)[1].item())
#                 W, H = a, b
#                 return (H, W)

#             if org.ndim == 2 and org.shape[-1] == 2:
#                 a = int(org[0, 0].item())
#                 b = int(org[0, 1].item())
#                 W, H = a, b
#                 return (H, W)

#             if org.ndim == 2 and org.shape[0] == 2:
#                 a = int(org[0, 0].item())
#                 b = int(org[1, 0].item())
#                 W, H = a, b
#                 return (H, W)

#             return tuple(batch["query_img"].shape[-2:])

#         try:
#             if isinstance(org, (list, tuple)) and len(org) >= 2:
#                 W = int(org[0])
#                 H = int(org[1])
#                 return (H, W)
#         except Exception:
#             pass

#         return tuple(batch["query_img"].shape[-2:])

#     def upsample_logit_mask(self, logit_mask, batch):
#         if self.training:
#             spatial_size = batch["query_img"].shape[-2:]
#         else:
#             spatial_size = self._get_org_hw(batch)
#         return F.interpolate(logit_mask, spatial_size, mode="bilinear", align_corners=True)

#     # -------------------------
#     # misc
#     # -------------------------
#     def train_mode(self):
#         self.train()
#         self.backbone.eval()

#     def configure_optimizers(self):
#         params = [p for p in self.parameters() if p.requires_grad]
#         return torch.optim.Adam(params, lr=self.args.lr)


