#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# OMP / MKL FIX — MUST BE FIRST (Windows safe)
# ============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# ============================================================

import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

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


@torch.no_grad()
def l2_norm(module: nn.Module) -> float:
    return float(torch.sqrt(sum((p ** 2).sum() for p in module.parameters())))


# ============================================================
# LoRA (minimal)
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scale


class ViTBackboneWithSlowLoRA(nn.Module):
    """
    timm ViT backbone frozen; LoRA inserted only in last_n_blocks MLP fc1/fc2.
    """
    def __init__(self,
                 model_name="vit_small_patch16_224",
                 pretrained=True,
                 lora_r=16,
                 lora_alpha=32,
                 last_n_blocks=4):
        super().__init__()
        import timm

        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.emb_dim = self.vit.num_features
        cfg = timm.data.resolve_model_data_config(self.vit)
        self.mean, self.std = cfg["mean"], cfg["std"]

        for p in self.vit.parameters():
            p.requires_grad = False

        for blk in self.vit.blocks[-last_n_blocks:]:
            for name in ["fc1", "fc2"]:
                layer = getattr(blk.mlp, name)
                setattr(blk.mlp, name, LoRALinear(layer, r=lora_r, alpha=lora_alpha))

        self.to(DEVICE)

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for m in self.modules():
            if isinstance(m, LoRALinear):
                yield m.A
                yield m.B

    def get_lora_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone()
                for k, v in self.state_dict().items()
                if ".A" in k or ".B" in k}

    def load_lora_state(self, sd: Dict[str, torch.Tensor]):
        self.load_state_dict(sd, strict=False)

    def forward(self, x):
        return self.vit(x)


# ============================================================
# Heads
# ============================================================
class LinearHead(nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, z):
        return self.fc(z)


# ============================================================
# Config
# ============================================================
@dataclass
class ClientCfg:
    cid: int
    num_classes: int = 100

    # optimization
    lr_fast: float = 2e-3
    lr_med: float = 2e-3
    lr_lora: float = 8e-4  # slow head + LoRA

    # local consolidation
    promote_k: int = 3
    promote_alpha: float = 0.3
    drift_th_local: float = 0.03  # stable vs full


# ============================================================
# Nested Client
# ============================================================
class NestedClient:
    def __init__(self, cfg: ClientCfg, backbone: ViTBackboneWithSlowLoRA):
        self.cfg = cfg
        self.backbone = backbone

        self.fast = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)
        self.med  = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)
        self.slow = LinearHead(backbone.emb_dim, cfg.num_classes).to(DEVICE)

        self.opt_fast = torch.optim.Adam(self.fast.parameters(), lr=cfg.lr_fast)
        self.opt_med  = torch.optim.Adam(self.med.parameters(),  lr=cfg.lr_med)
        self.opt_slow = torch.optim.Adam(
            list(self.slow.parameters()) + list(self.backbone.lora_parameters()),
            lr=cfg.lr_lora
        )

        self.stable_fast = 0
        self.stable_med = 0

    # ---- Federated IO ----
    def get_med(self):  return to_cpu(self.med.state_dict())
    def get_slow(self): return to_cpu(self.slow.state_dict())
    def get_lora(self): return self.backbone.get_lora_state()
    def set_med(self, sd):  self.med.load_state_dict(sd)
    def set_slow(self, sd): self.slow.load_state_dict(sd)
    def set_lora(self, sd): self.backbone.load_lora_state(sd)

    # ---- logits ----
    def logits_full(self, z):
        return self.fast(z) + self.med(z) + self.slow(z)

    def logits_stable(self, z):
        return self.med(z) + self.slow(z)

    @torch.no_grad()
    def local_drift(self, anchor_batch: torch.Tensor) -> float:
        """
        local drift score = mean|softmax(stable)-softmax(full)| on local anchor batch.
        (Only debug / local consolidation signal; not sent to server.)
        """
        self.backbone.eval()
        self.fast.eval(); self.med.eval(); self.slow.eval()

        x = anchor_batch.to(DEVICE)
        z = self.backbone(x)
        p_stable = F.softmax(self.logits_stable(z), -1)
        p_full   = F.softmax(self.logits_full(z), -1)
        return float((p_stable - p_full).abs().mean().item())

    @torch.no_grad()
    def signature_on_reference_A(self, refA_loader) -> torch.Tensor:
        """
        DATA-FREE SERVER: return s_i in R^C on CANONICAL A reference set.
        Same input distribution for all clients.
        """
        self.backbone.eval()
        self.med.eval(); self.slow.eval()

        acc = None
        n = 0
        for x, _ in refA_loader:
            x = x.to(DEVICE)
            z = self.backbone(x)
            p = F.softmax(self.logits_stable(z), -1).detach().cpu()  # [B,C]
            acc = p.sum(dim=0) if acc is None else (acc + p.sum(dim=0))
            n += p.shape[0]

        if acc is None:
            acc = torch.zeros(self.cfg.num_classes)
            n = 1
        return acc / max(1, n)  # [C]

    def train_one_round(self,
                        train_loader,
                        anchor_batch_for_local_drift: Optional[torch.Tensor],
                        refA_loader_for_signature,
                        allow_med_to_slow: bool):
        """
        allow_med_to_slow is decided by SERVER gate for this round.
        """
        self.backbone.train()
        self.fast.train(); self.med.train(); self.slow.train()

        losses = []
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

        d_local = 0.0
        if anchor_batch_for_local_drift is not None:
            d_local = self.local_drift(anchor_batch_for_local_drift)

        # local stability counters
        if d_local < self.cfg.drift_th_local:
            self.stable_fast += 1
            self.stable_med += 1
        else:
            self.stable_fast = 0
            self.stable_med = 0

        promoted = {"f2m": 0, "m2s": 0, "blocked": 0}

        # fast -> med always allowed
        if self.stable_fast >= self.cfg.promote_k:
            for pm, pf in zip(self.med.parameters(), self.fast.parameters()):
                pm.data = (1 - self.cfg.promote_alpha) * pm.data + self.cfg.promote_alpha * pf.data
            self.stable_fast = 0
            promoted["f2m"] = 1

        # med -> slow server-gated
        if self.stable_med >= self.cfg.promote_k + 2:
            if allow_med_to_slow:
                for ps, pm in zip(self.slow.parameters(), self.med.parameters()):
                    ps.data = (1 - self.cfg.promote_alpha) * ps.data + self.cfg.promote_alpha * pm.data
                self.stable_med = 0
                promoted["m2s"] = 1
            else:
                promoted["blocked"] = 1

        # IMPORTANT: signature computed AFTER training, on canonical A reference set
        sigA = self.signature_on_reference_A(refA_loader_for_signature)

        dbg = {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "localDrift": float(d_local),
            "norm_fast": l2_norm(self.fast),
            "norm_med": l2_norm(self.med),
            "norm_slow": l2_norm(self.slow),
        }
        return dbg, promoted, sigA

    @torch.no_grad()
    def eval_acc(self, loader, use_fast: bool):
        self.backbone.eval()
        self.fast.eval(); self.med.eval(); self.slow.eval()

        correct, total = 0, 0
        for x, y in loader:
            x = x.to(DEVICE)
            z = self.backbone(x)
            logits = self.logits_full(z) if use_fast else self.logits_stable(z)
            pred = logits.argmax(-1).cpu()
            correct += (pred == y).sum().item()
            total += y.numel()
        return correct / max(1, total)
