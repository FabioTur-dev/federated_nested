#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# OMP / MKL FIX — MUST BE FIRST
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
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import ImageOps

from nested_client import seed_all, ViTBackboneWithSlowLoRA, ClientCfg, NestedClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# FedAvg
# ============================================================
def fedavg(sd_list):
    out = {}
    for k in sd_list[0]:
        out[k] = sum(sd[k] for sd in sd_list) / len(sd_list)
    return out


# ============================================================
# Deterministic client-specific B (killer) — ONLY for training
# ============================================================
class ClientSpecificB:
    def __init__(self, cid: int):
        rng = random.Random(10_000 + cid)
        self.brightness = 0.55 + 0.10 * rng.random()
        self.contrast   = 1.55 + 0.25 * rng.random()
        self.saturation = 0.35 + 0.25 * rng.random()
        self.solarize_thr = 96 + int(64 * rng.random())   # [96..160]
        self.posterize_bits = 3 if rng.random() < 0.5 else 4

        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        self.perm = perms[cid % len(perms)]

        k = 7 if (cid % 2 == 0) else 9
        sigma = 1.8 + 0.8 * (cid % 3)
        self.blur = transforms.GaussianBlur(kernel_size=k, sigma=sigma)

    def __call__(self, img):
        img = TF.adjust_brightness(img, self.brightness)
        img = TF.adjust_contrast(img, self.contrast)
        img = TF.adjust_saturation(img, self.saturation)
        img = ImageOps.solarize(img, threshold=self.solarize_thr)
        img = ImageOps.posterize(img, bits=self.posterize_bits)

        t = TF.to_tensor(img)
        t = t[list(self.perm), :, :]
        img = TF.to_pil_image(t)

        img = self.blur(img)
        return img


def make_transform_A(mean, std, train: bool):
    ops = [transforms.Resize(224)]
    if train:
        ops += [transforms.RandomHorizontalFlip(p=0.5)]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def make_transform_train(regime: str, mean, std, cid: int):
    """
    Regime A: mild
    Regime B: killer shift depends on client id (domain heterogeneity)
    """
    ops = [transforms.Resize(224)]
    if regime == "A":
        ops += [transforms.RandomHorizontalFlip(p=0.5)]
    elif regime == "B":
        ops += [ClientSpecificB(cid)]
        ops += [transforms.RandomHorizontalFlip(p=0.2)]
    else:
        raise ValueError(regime)
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


# ============================================================
# JS divergence gate on canonical A reference signatures
# ============================================================
@torch.no_grad()
def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # p,q: [C]
    p = (p + eps) / (p + eps).sum(dim=-1, keepdim=False)
    q = (q + eps) / (q + eps).sum(dim=-1, keepdim=False)
    m = 0.5 * (p + q)
    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


@torch.no_grad()
def server_disagreement_js(sig_list: list[torch.Tensor]) -> float:
    """
    sig_list: list of [C] probs on SAME reference A (client-side).
    returns mean JS divergence to the mean signature.
    """
    S = torch.stack(sig_list, dim=0)  # [M,C]
    mean = S.mean(dim=0)              # [C]
    ds = torch.stack([js_divergence(S[i], mean) for i in range(S.shape[0])], dim=0)
    return float(ds.mean().item())


# ============================================================
# Main
# ============================================================
def main():
    seed_all(0)

    # ---- knobs ----
    NUM_CLIENTS = 10
    CLIENT_FRAC = 0.5
    TOTAL_ROUNDS = 30
    SLOW_PERIOD = 10

    # gate threshold (JS is small; tune if needed)
    # target: ~0 in early A; >th when client B diverges
    DRIFT_TH_SERVER = 0.0015

    # schedule A->B->A
    schedule = (["A"] * 10) + (["B"] * 10) + (["A"] * 10)

    # init backbone template for normalization
    template = ViTBackboneWithSlowLoRA()
    mean, std = template.mean, template.std

    # raw train for splitting only
    train_raw = datasets.CIFAR100("./data", train=True, download=True)
    idxs = np.arange(len(train_raw))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, NUM_CLIENTS)

    # reference set (shared, stored locally on clients, server never sees images)
    REF_SIZE = 256
    ref_indices = list(idxs[:REF_SIZE])

    # per-client train indices exclude reference
    ref_set = set(ref_indices)
    client_train_idxs = []
    for i in range(NUM_CLIENTS):
        client_train_idxs.append([j for j in splits[i].tolist() if j not in ref_set])

    # async offsets (same as your runs)
    rng = random.Random(123)
    offsets = [rng.randint(0, 6) for _ in range(NUM_CLIENTS)]
    print(f"[INFO] Async offsets per client (rounds): {offsets}")

    # canonical datasets for ref/test (A only)
    tfm_refA = make_transform_A(mean, std, train=False)
    ref_ds = datasets.CIFAR100("./data", train=True, download=False, transform=tfm_refA)
    ref_loader = DataLoader(Subset(ref_ds, ref_indices), batch_size=64, shuffle=False, num_workers=0)

    testA = datasets.CIFAR100("./data", train=False, download=False, transform=tfm_refA)
    test_loaderA = DataLoader(testA, batch_size=256, shuffle=False, num_workers=0)

    # for reporting B accuracy, use a fixed “canonical B0” (not client-specific) – just for plots
    tfm_testB0 = transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.4, contrast=0.6, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    testB0 = datasets.CIFAR100("./data", train=False, download=False, transform=tfm_testB0)
    test_loaderB0 = DataLoader(testB0, batch_size=256, shuffle=False, num_workers=0)

    # clients
    clients = [NestedClient(ClientCfg(i), ViTBackboneWithSlowLoRA()) for i in range(NUM_CLIENTS)]

    # init globals
    g_med = clients[0].get_med()
    g_slow = clients[0].get_slow()
    g_lora = clients[0].get_lora()

    print("\n===== NESTED KILLER (JS GATE on CANONICAL A-REF) =====\n")
    print(f"[INFO] Gate: freeze slow NEXT round if JS_disagree(A-ref) > {DRIFT_TH_SERVER:.4f}")
    print(f"[INFO] SLOW_PERIOD={SLOW_PERIOD}, REF_SIZE={REF_SIZE}\n")

    freeze_slow_now = False

    for r in range(TOTAL_ROUNDS):
        reg_global = schedule[r]
        sel = random.sample(range(NUM_CLIENTS), max(1, int(NUM_CLIENTS * CLIENT_FRAC)))

        # broadcast globals
        for cid in sel:
            clients[cid].set_med(g_med)
            clients[cid].set_slow(g_slow)
            clients[cid].set_lora(g_lora)

        sigs = []
        losses = []
        local_drifts = []
        regs_c = []
        promo = {"f2m": 0, "m2s": 0, "blocked": 0}

        for cid in sel:
            rc = (r + offsets[cid]) % len(schedule)
            reg_c = schedule[rc]
            regs_c.append(reg_c)

            tfm_train = make_transform_train(reg_c, mean, std, cid=cid)
            ds_train = datasets.CIFAR100("./data", train=True, download=False, transform=tfm_train)
            loader_train = DataLoader(Subset(ds_train, client_train_idxs[cid]),
                                      batch_size=64, shuffle=True, num_workers=0)

            # local drift anchor batch: first batch from canonical A ref
            anchor_batch = next(iter(ref_loader))[0]  # tensor batch

            dbg, prom, sigA = clients[cid].train_one_round(
                train_loader=loader_train,
                anchor_batch_for_local_drift=anchor_batch,
                refA_loader_for_signature=ref_loader,
                allow_med_to_slow=(not freeze_slow_now)
            )

            losses.append(dbg["loss"])
            local_drifts.append(dbg["localDrift"])
            sigs.append(sigA)

            promo["f2m"] += prom["f2m"]
            promo["m2s"] += prom["m2s"]
            promo["blocked"] += prom["blocked"]

        # aggregate med every round
        g_med = fedavg([clients[c].get_med() for c in sel])

        # aggregate slow/LoRA only if period hits AND not frozen
        did_slow_agg = 0
        if (r + 1) % SLOW_PERIOD == 0 and (not freeze_slow_now):
            g_slow = fedavg([clients[c].get_slow() for c in sel])
            g_lora = fedavg([clients[c].get_lora() for c in sel])
            did_slow_agg = 1

        # disagreement AFTER training, on canonical A-ref signatures
        d_js = server_disagreement_js(sigs)
        freeze_slow_next = (d_js > DRIFT_TH_SERVER)

        # evaluate global stability (med+slow) with a fresh evaluator
        evaluator = NestedClient(ClientCfg(999), ViTBackboneWithSlowLoRA())
        evaluator.set_med(g_med)
        evaluator.set_slow(g_slow)
        evaluator.set_lora(g_lora)
        accA = evaluator.eval_acc(test_loaderA, use_fast=False)
        accB = evaluator.eval_acc(test_loaderB0, use_fast=False)

        tag = ""
        if r > 0 and schedule[r - 1] == "B" and schedule[r] == "A":
            tag = "  <== RECOVERY POINT (B->A)"

        print(
            f"[R{r:02d} | global={reg_global} | sel={len(sel)} | slow_agg={did_slow_agg} | "
            f"freeze={int(freeze_slow_now)}→{int(freeze_slow_next)}] "
            f"clientRegs={''.join(regs_c)} | "
            f"loss={np.mean(losses):.3f} | localDrift={np.mean(local_drifts):.3f} | "
            f"JS(A-ref)={d_js:.4f} | "
            f"promo(f→m={promo['f2m']}, m→s={promo['m2s']}, blocked={promo['blocked']}) | "
            f"GlobalAcc(A={accA:.3f}, B0={accB:.3f}){tag}"
        )

        freeze_slow_now = freeze_slow_next

    print("\nDone.\n")


if __name__ == "__main__":
    main()
