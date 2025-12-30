#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================= OMP SAFE =================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# ============================================

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_cpu(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


# ================= LoRA ======================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.scale = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


class ViTBackboneWithSlowLoRA(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", pretrained=True):
        super().__init__()
        import timm
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.emb_dim = self.vit.num_features
        cfg = timm.data.resolve_model_data_config(self.vit)
        self.mean, self.std = cfg["mean"], cfg["std"]

        for p in self.vit.parameters():
            p.requires_grad = False

        for blk in self.vit.blocks[-4:]:
            blk.mlp.fc1 = LoRALinear(blk.mlp.fc1)
            blk.mlp.fc2 = LoRALinear(blk.mlp.fc2)

        self.to(DEVICE)

    def lora_parameters(self):
        for m in self.modules():
            if isinstance(m, LoRALinear):
                yield m.A
                yield m.B

    def get_lora(self):
        return {k: v.cpu() for k, v in self.state_dict().items() if ".A" in k or ".B" in k}

    def set_lora(self, sd):
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        return self.vit(x)


class LinearHead(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, z):
        return self.fc(z)


@dataclass
class ClientCfg:
    cid: int
    num_classes: int = 100
    lr_fast: float = 2e-3
    lr_med: float = 2e-3
    lr_slow: float = 8e-4
    promote_k: int = 3
    promote_alpha: float = 0.3
    drift_th_local: float = 0.03


class NestedClient:
    def __init__(self, cfg: ClientCfg, backbone):
        self.cfg = cfg
        self.backbone = backbone

        self.fast = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)
        self.med  = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)
        self.slow = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)

        self.opt_fast = torch.optim.Adam(self.fast.parameters(), lr=cfg.lr_fast)
        self.opt_med  = torch.optim.Adam(self.med.parameters(),  lr=cfg.lr_med)
        self.opt_slow = torch.optim.Adam(
            list(self.slow.parameters()) + list(self.backbone.lora_parameters()),
            lr=cfg.lr_slow
        )

        self.stable_fast = 0
        self.stable_med = 0

    # -------- federation IO --------
    def get_med(self): return to_cpu(self.med.state_dict())
    def get_slow(self): return to_cpu(self.slow.state_dict())
    def get_lora(self): return self.backbone.get_lora()
    def set_med(self, sd): self.med.load_state_dict(sd)
    def set_slow(self, sd): self.slow.load_state_dict(sd)
    def set_lora(self, sd): self.backbone.set_lora(sd)

    # -------- logits --------
    def logits_full(self, z): return self.fast(z) + self.med(z) + self.slow(z)
    def logits_stable(self, z): return self.med(z) + self.slow(z)

    @torch.no_grad()
    def compute_signature(self, loader):
        acc = torch.zeros(self.cfg.num_classes)
        n = 0
        self.backbone.eval()
        self.med.eval(); self.slow.eval()
        for x, _ in loader:
            x = x.to(DEVICE)
            z = self.backbone(x)
            p = F.softmax(self.logits_stable(z), -1).cpu()
            acc += p.sum(0)
            n += p.shape[0]
        return acc / max(1, n)

    def train_round(self, train_loader, anchor_batch, refA_loader, refB_loader, allow_slow):
        losses = []
        self.backbone.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            z = self.backbone(x)
            loss = F.cross_entropy(self.logits_full(z), y)
            self.opt_fast.zero_grad()
            self.opt_med.zero_grad()
            self.opt_slow.zero_grad()
            loss.backward()
            self.opt_fast.step()
            self.opt_med.step()
            self.opt_slow.step()
            losses.append(loss.item())

        # local drift
        with torch.no_grad():
            z = self.backbone(anchor_batch.to(DEVICE))
            p_full = F.softmax(self.logits_full(z), -1)
            p_stab = F.softmax(self.logits_stable(z), -1)
            d = (p_full - p_stab).abs().mean().item()

        if d < self.cfg.drift_th_local:
            self.stable_fast += 1
            self.stable_med += 1
        else:
            self.stable_fast = self.stable_med = 0

        promo = {"f2m": 0, "m2s": 0, "blocked": 0}

        if self.stable_fast >= self.cfg.promote_k:
            for pm, pf in zip(self.med.parameters(), self.fast.parameters()):
                pm.data = (1 - self.cfg.promote_alpha) * pm + self.cfg.promote_alpha * pf
            self.stable_fast = 0
            promo["f2m"] = 1

        if self.stable_med >= self.cfg.promote_k + 2:
            if allow_slow:
                for ps, pm in zip(self.slow.parameters(), self.med.parameters()):
                    ps.data = (1 - self.cfg.promote_alpha) * ps + self.cfg.promote_alpha * pm
                promo["m2s"] = 1
                self.stable_med = 0
            else:
                promo["blocked"] = 1

        sigA = self.compute_signature(refA_loader)
        sigB = self.compute_signature(refB_loader)

        return np.mean(losses), d, promo, sigA, sigB

    @torch.no_grad()
    def eval_acc(self, loader, use_fast=False):
        self.backbone.eval()
        correct, total = 0, 0
        for x, y in loader:
            x = x.to(DEVICE)
            z = self.backbone(x)
            logits = self.logits_full(z) if use_fast else self.logits_stable(z)
            pred = logits.argmax(-1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)
