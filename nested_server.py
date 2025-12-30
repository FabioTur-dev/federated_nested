#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from nested_client import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============== JS divergence ===============
@torch.no_grad()
def js_div(p, q, eps=1e-8):
    p = (p + eps) / (p.sum() + eps)
    q = (q + eps) / (q.sum() + eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum())


def server_disagree(sigA_list, sigB_list):
    S_A = torch.stack(sigA_list)
    S_B = torch.stack(sigB_list)
    mA = S_A.mean(0)
    mB = S_B.mean(0)
    dA = torch.stack([js_div(s, mA) for s in S_A]).mean()
    dB = torch.stack([js_div(s, mB) for s in S_B]).mean()
    return float(dA), float(dB)


def fedavg(sds):
    return {k: sum(sd[k] for sd in sds) / len(sds) for k in sds[0]}


def main():
    seed_all(0)

    NUM_CLIENTS = 10
    CLIENT_FRAC = 0.5
    TOTAL_ROUNDS = 30
    SLOW_PERIOD = 10
    TH_SERVER = 0.0008

    schedule = ["A"] * 10 + ["B"] * 10 + ["A"] * 10

    backbone = ViTBackboneWithSlowLoRA()
    norm = (backbone.mean, backbone.std)

    train_raw = datasets.CIFAR100("./data", train=True, download=True)
    idxs = np.random.permutation(len(train_raw))
    splits = np.array_split(idxs, NUM_CLIENTS)

    REF_IDX = idxs[:128].tolist()

    clients = [NestedClient(ClientCfg(i), ViTBackboneWithSlowLoRA()) for i in range(NUM_CLIENTS)]

    g_med = clients[0].get_med()
    g_slow = clients[0].get_slow()
    g_lora = clients[0].get_lora()

    freeze_now = False

    print("\n===== NESTED KILLER (JS + DOUBLE SIGNATURE) =====\n")

    for r in range(TOTAL_ROUNDS):
        sel = random.sample(range(NUM_CLIENTS), int(NUM_CLIENTS * CLIENT_FRAC))
        global_reg = schedule[r]

        for c in sel:
            clients[c].set_med(g_med)
            clients[c].set_slow(g_slow)
            clients[c].set_lora(g_lora)

        sigA, sigB = [], []
        losses, drifts = [], []
        promo = {"f2m": 0, "m2s": 0, "blocked": 0}

        for c in sel:
            reg_c = schedule[(r + c) % len(schedule)]

            tfm_train = transforms.Compose([
                transforms.Resize(224),
                transforms.ColorJitter(brightness=0.4 * (c + 1) / NUM_CLIENTS),
                transforms.ToTensor(),
                transforms.Normalize(*norm)
            ])

            tfm_A = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(*norm)
            ])

            tfm_B = transforms.Compose([
                transforms.Resize(224),
                transforms.ColorJitter(contrast=0.6 * (c + 1) / NUM_CLIENTS),
                transforms.ToTensor(),
                transforms.Normalize(*norm)
            ])

            ds_train = datasets.CIFAR100("./data", train=True, transform=tfm_train)
            ds_refA = datasets.CIFAR100("./data", train=True, transform=tfm_A)
            ds_refB = datasets.CIFAR100("./data", train=True, transform=tfm_B)

            train_loader = DataLoader(Subset(ds_train, splits[c]), batch_size=64, shuffle=True)
            refA_loader = DataLoader(Subset(ds_refA, REF_IDX), batch_size=64)
            refB_loader = DataLoader(Subset(ds_refB, REF_IDX), batch_size=64)

            anchor = next(iter(refA_loader))[0]

            loss, d, p, sA, sB = clients[c].train_round(
                train_loader, anchor, refA_loader, refB_loader, not freeze_now
            )

            losses.append(loss)
            drifts.append(d)
            sigA.append(sA)
            sigB.append(sB)
            for k in promo:
                promo[k] += p[k]

        g_med = fedavg([clients[c].get_med() for c in sel])
        slow_agg = 0
        if (r + 1) % SLOW_PERIOD == 0 and not freeze_now:
            g_slow = fedavg([clients[c].get_slow() for c in sel])
            g_lora = fedavg([clients[c].get_lora() for c in sel])
            slow_agg = 1

        dA, dB = server_disagree(sigA, sigB)
        freeze_next = max(dA, dB) > TH_SERVER

        evaluator = NestedClient(ClientCfg(999), ViTBackboneWithSlowLoRA())
        evaluator.set_med(g_med)
        evaluator.set_slow(g_slow)
        evaluator.set_lora(g_lora)

        testA = DataLoader(datasets.CIFAR100("./data", train=False, transform=tfm_A), batch_size=256)
        testB = DataLoader(datasets.CIFAR100("./data", train=False, transform=tfm_B), batch_size=256)

        accA = evaluator.eval_acc(testA)
        accB = evaluator.eval_acc(testB)

        print(
            f"[R{r:02d} | G={global_reg} | slowAgg={slow_agg} | "
            f"freeze={int(freeze_now)}→{int(freeze_next)} | "
            f"loss={np.mean(losses):.3f} | drift={np.mean(drifts):.3f} | "
            f"JS(A)={dA:.4f} JS(B)={dB:.4f} | "
            f"promo f2m={promo['f2m']} m2s={promo['m2s']} blocked={promo['blocked']} | "
            f"Acc(A={accA:.3f}, B={accB:.3f})"
        )

        freeze_now = freeze_next

    print("\nDone.\n")


if __name__ == "__main__":
    main()
