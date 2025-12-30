#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import ImageOps

from nested_client import seed_all, ViTBackboneWithSlowLoRA, ClientCfg, NestedClient


# =====================================================
# Utils
# =====================================================
def fedavg(sd_list):
    out = {}
    for k in sd_list[0]:
        out[k] = sum(sd[k] for sd in sd_list) / len(sd_list)
    return out


@torch.no_grad()
def js_divergence(p, q, eps=1e-8):
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum())


@torch.no_grad()
def server_disagreement_js(sig_list):
    S = torch.stack(sig_list, dim=0)
    mean = S.mean(dim=0)
    return float(torch.stack([js_divergence(s, mean) for s in S]).mean())


# =====================================================
# Drift (killer) — IDENTICO alla tua versione
# =====================================================
class StrongDrift:
    def __init__(self, seed):
        rng = random.Random(seed)
        self.brightness = 0.6 + 0.3 * rng.random()
        self.contrast = 2.0 + 0.5 * rng.random()
        self.saturation = 0.3 + 0.4 * rng.random()
        self.blur = transforms.GaussianBlur(11, sigma=3.0)

    def __call__(self, img):
        img = TF.adjust_brightness(img, self.brightness)
        img = TF.adjust_contrast(img, self.contrast)
        img = TF.adjust_saturation(img, self.saturation)
        img = ImageOps.solarize(img, threshold=64)
        img = ImageOps.posterize(img, bits=3)
        return self.blur(img)


class MildDrift:
    def __init__(self, seed):
        self.jitter = transforms.ColorJitter(0.15, 0.2, 0.15)

    def __call__(self, img):
        return self.jitter(img)


class MovingLabelConditionalB:
    def __init__(self, cid, swap_period):
        self.cid = cid
        self.group = cid % 2
        self.swap_period = swap_period
        self.strong = StrongDrift(10_000 + cid)
        self.mild = MildDrift(20_000 + cid)

    def __call__(self, img, y, global_round):
        phase = global_round // self.swap_period
        low = y <= 49
        if phase % 2 == 0:
            strong = (low if self.group == 0 else not low)
        else:
            strong = (not low if self.group == 0 else low)
        return (self.strong(img) if strong else self.mild(img)), strong, phase


class CIFAR100NestedWrapper(Dataset):
    def __init__(self, base, indices, mean, std, cid, regime, global_round, swap_period):
        self.base = base
        self.indices = indices
        self.cid = cid
        self.regime = regime
        self.global_round = global_round
        self.resize = transforms.Resize(224)
        self.norm = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()
        self.movingB = MovingLabelConditionalB(cid, swap_period)
        self.strong = 0
        self.total = 0
        self.phase = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        img = self.resize(img)
        used = False
        if self.regime == "B":
            img, used, self.phase = self.movingB(img, y, self.global_round)
        self.total += 1
        if used:
            self.strong += 1
        x = self.norm(self.to_tensor(img))
        return x, y

    def strong_ratio(self):
        return self.strong / max(1, self.total)


# =====================================================
# MAIN
# =====================================================
def main():
    seed_all(0)

    NUM_CLIENTS = 10
    CLIENT_FRAC = 0.5
    TOTAL_ROUNDS = 30
    SLOW_PERIOD = 10

    # 🔥 GATE PARAMS (FIXED)
    GATE_WARMUP = 3
    BASELINE_WIN = 4          # <<< FIX
    GATE_K = 3.0
    MAJORITY_A = 0.8          # <<< FIX

    SWAP_PERIOD = 2
    schedule = ["A"] * 10 + ["B"] * 10 + ["A"] * 10

    backbone_template = ViTBackboneWithSlowLoRA()
    mean, std = backbone_template.mean, backbone_template.std

    train_base = datasets.CIFAR100("./data", train=True, download=True, transform=None)

    idxs = np.arange(len(train_base))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, NUM_CLIENTS)

    ref_idx = idxs[:256].tolist()
    ref_ds = datasets.CIFAR100("./data", train=True, transform=transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]))
    ref_loader = DataLoader(Subset(ref_ds, ref_idx), batch_size=64)

    clients = [NestedClient(ClientCfg(i), ViTBackboneWithSlowLoRA()) for i in range(NUM_CLIENTS)]
    g_med = clients[0].get_med()
    g_slow = clients[0].get_slow()
    g_lora = clients[0].get_lora()

    offsets = [random.randint(0, 6) for _ in range(NUM_CLIENTS)]
    print(f"[INFO] Async offsets per client (rounds): {offsets}")

    print("\n===== NESTED EXTREME MOVING KILLER (FINAL) =====\n")

    baseline_js = []
    baseline_frozen = False
    mu = sd = 0.0
    th = float("inf")
    freeze_now = False

    for r in range(TOTAL_ROUNDS):
        reg_global = schedule[r]
        sel = random.sample(range(NUM_CLIENTS), max(1, int(NUM_CLIENTS * CLIENT_FRAC)))
        regs = []

        sigs = []
        losses = []
        strong_ratios = []

        for cid in sel:
            clients[cid].set_med(g_med)
            clients[cid].set_slow(g_slow)
            clients[cid].set_lora(g_lora)

        for cid in sel:
            rc = (r + offsets[cid]) % len(schedule)
            reg = schedule[rc]
            regs.append(reg)

            ds = CIFAR100NestedWrapper(
                train_base,
                splits[cid].tolist(),
                mean, std,
                cid, reg, r, SWAP_PERIOD
            )
            loader = DataLoader(ds, batch_size=64, shuffle=True)

            dbg, prom, sig = clients[cid].train_one_round(
                loader,
                next(iter(ref_loader))[0],
                ref_loader,
                allow_med_to_slow=not freeze_now
            )
            sigs.append(sig)
            losses.append(dbg["loss"])
            strong_ratios.append(ds.strong_ratio())

        g_med = fedavg([clients[c].get_med() for c in sel])
        if (r + 1) % SLOW_PERIOD == 0 and not freeze_now:
            g_slow = fedavg([clients[c].get_slow() for c in sel])
            g_lora = fedavg([clients[c].get_lora() for c in sel])

        js = server_disagreement_js(sigs)
        frac_A = sum(c == "A" for c in regs) / len(regs)

        # 🔥 BASELINE COLLECTION (FIXED)
        if not baseline_frozen:
            if r >= GATE_WARMUP and reg_global == "A" and frac_A >= MAJORITY_A:
                baseline_js.append(js)
                if len(baseline_js) >= BASELINE_WIN:
                    mu = float(np.mean(baseline_js))
                    sd = float(np.std(baseline_js) + 1e-9)
                    th = mu + GATE_K * sd
                    baseline_frozen = True
                    print(f"    [BASELINE FROZEN] mu={mu:.4f}, sd={sd:.4f}, th={th:.4f}")

        freeze_next = freeze_now
        if baseline_frozen and r >= GATE_WARMUP:
            freeze_next = js > th

        print(
            f"[R{r:02d} | global={reg_global} | freeze={int(freeze_now)}→{int(freeze_next)} | "
            f"baseFrozen={int(baseline_frozen)} | "
            f"JS={js:.4f} | th={th:.4f} | "
            f"fracA={frac_A:.2f} | strongRatio={np.mean(strong_ratios):.2f}]"
        )

        if baseline_frozen:
            if freeze_next and not freeze_now:
                print(f"    [GATE-ON] JS {js:.4f} > th {th:.4f}")
            if not freeze_next and freeze_now:
                print(f"    [GATE-OFF] JS {js:.4f} <= th {th:.4f}")

        freeze_now = freeze_next

    print("\nDone.\n")


if __name__ == "__main__":
    main()
