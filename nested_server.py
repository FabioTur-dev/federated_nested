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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import ImageOps, ImageEnhance

from nested_client import seed_all, ViTBackboneWithSlowLoRA, ClientCfg, NestedClient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fedavg(sd_list):
    out = {}
    for k in sd_list[0]:
        out[k] = sum(sd[k] for sd in sd_list) / len(sd_list)
    return out


# ============================================================
# KILLER DRIFT: Client-specific B transform
# (simulates per-site camera calibration / filters / lighting)
# ============================================================
class ClientSpecificB:
    def __init__(self, cid: int):
        # deterministic params from cid
        # (ensure different clients see different "B" worlds)
        rng = random.Random(10_000 + cid)
        self.brightness = 0.55 + 0.10 * rng.random()
        self.contrast   = 1.55 + 0.25 * rng.random()
        self.saturation = 0.35 + 0.25 * rng.random()
        self.gamma      = 0.70 + 0.60 * rng.random()
        self.solarize_thr = 96 + int(64 * rng.random())  # [96..160]
        self.posterize_bits = 3 if rng.random() < 0.5 else 4

        # channel permutation (strong, but label-preserving)
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        self.perm = perms[cid % len(perms)]

        # blur varies by client
        k = 7 if (cid % 2 == 0) else 9
        sigma = 1.8 + 0.8 * (cid % 3)
        self.blur = transforms.GaussianBlur(kernel_size=k, sigma=sigma)

    def __call__(self, img):
        # deterministic “style” pipeline
        img = TF.adjust_brightness(img, self.brightness)
        img = TF.adjust_contrast(img, self.contrast)
        img = TF.adjust_saturation(img, self.saturation)

        # gamma via PIL enhancer
        img = TF.to_pil_image(TF.to_tensor(img))
        img = ImageEnhance.Brightness(img).enhance(1.0)  # no-op but keeps PIL path stable
        # apply gamma on tensor
        t = TF.to_tensor(img).clamp(0, 1)
        t = t.pow(self.gamma)
        img = TF.to_pil_image(t)

        # strong but label-preserving operations
        img = ImageOps.solarize(img, threshold=self.solarize_thr)
        img = ImageOps.posterize(img, bits=self.posterize_bits)

        # channel permutation
        t = TF.to_tensor(img)
        t = t[list(self.perm), :, :]
        img = TF.to_pil_image(t)

        img = self.blur(img)
        return img


def make_transform(regime: str, norm, train: bool, cid: int = 0):
    mean, std = norm
    ops = [transforms.Resize(224)]

    if regime == "A":
        if train:
            ops += [transforms.RandomHorizontalFlip(p=0.5)]
    elif regime == "B":
        # killer: different B per client
        ops += [ClientSpecificB(cid)]
        if train:
            # keep deterministic shift; you can still flip to add mild variability
            ops += [transforms.RandomHorizontalFlip(p=0.2)]
    else:
        raise ValueError(regime)

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


@torch.no_grad()
def server_disagreement_from_signatures(sig_list: list[torch.Tensor]) -> float:
    S = torch.stack(sig_list, dim=0)  # [M,C]
    S_mean = S.mean(dim=0, keepdim=True)
    return float((S - S_mean).abs().mean().item())


def main():
    seed_all(0)

    # ============================================================
    # EXPERIMENT CONFIG (killer)
    # ============================================================
    NUM_CLIENTS = 10
    CLIENT_FRAC = 0.5
    TOTAL_ROUNDS = 30

    # make slow harder to corrupt; and let gate matter
    SLOW_PERIOD = 10  # rarer slow aggregation => more sensitive to gate
    ASYNC_MAX_OFFSET = 6

    # tighter threshold: signatures are averaged vectors
    DRIFT_TH_SERVER = 0.004

    # schedule A->B->A
    schedule = (["A"] * 10) + (["B"] * 10) + (["A"] * 10)

    # template for norm
    template = ViTBackboneWithSlowLoRA()
    norm = (template.mean, template.std)

    # raw train for splitting only
    train_raw = datasets.CIFAR100("./data", train=True, download=True)

    # deterministic test sets (global A/B)
    testA = datasets.CIFAR100("./data", train=False, download=False,
                              transform=make_transform("A", norm, train=False, cid=0))
    # for B test we choose cid=0 as “canonical B”; this is fine for plotting,
    # but per-client B differs. (That's the point.)
    testB = datasets.CIFAR100("./data", train=False, download=False,
                              transform=make_transform("B", norm, train=False, cid=0))
    loaderA = DataLoader(testA, batch_size=256, shuffle=False, num_workers=0)
    loaderB = DataLoader(testB, batch_size=256, shuffle=False, num_workers=0)

    # split indices per client
    idxs = np.arange(len(train_raw))
    np.random.shuffle(idxs)
    splits = np.array_split(idxs, NUM_CLIENTS)

    # REFERENCE SET (shared, public/unlabeled)
    # Replicated on clients (server does not see images; in simulation we build it here)
    REF_SIZE = 512
    ref_indices = list(idxs[:REF_SIZE])

    # per-client train indices (exclude reference indices)
    ref_set = set(ref_indices)
    client_train_idxs = []
    for i in range(NUM_CLIENTS):
        client_train_idxs.append([j for j in splits[i].tolist() if j not in ref_set])

    # async offsets
    rng = random.Random(123)
    offsets = [rng.randint(0, ASYNC_MAX_OFFSET) for _ in range(NUM_CLIENTS)]
    print(f"[INFO] Async offsets per client (rounds): {offsets}")

    # clients
    clients = [NestedClient(ClientCfg(i), ViTBackboneWithSlowLoRA()) for i in range(NUM_CLIENTS)]

    # init globals
    g_med = clients[0].get_med()
    g_slow = clients[0].get_slow()
    g_lora = clients[0].get_lora()

    print("\n===== NESTED KILLER RUN (CLIENT-SPECIFIC B + DATA-FREE SERVER GATE) =====\n")
    print(f"[INFO] SLOW_PERIOD={SLOW_PERIOD}, DRIFT_TH_SERVER={DRIFT_TH_SERVER:.4f}\n")

    freeze_slow_now = False
    last_freeze = None

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

            # client train transform depends on regime AND cid (killer for B)
            tfm_train = make_transform(reg_c, norm, train=True, cid=cid)
            ds_train = datasets.CIFAR100("./data", train=True, download=False, transform=tfm_train)
            loader_train = DataLoader(Subset(ds_train, client_train_idxs[cid]),
                                      batch_size=64, shuffle=True, num_workers=0)

            # shared reference set: ALWAYS A deterministic, same indices for everyone
            tfm_ref = make_transform("A", norm, train=False, cid=0)
            ds_ref = datasets.CIFAR100("./data", train=True, download=False, transform=tfm_ref)
            loader_ref = DataLoader(Subset(ds_ref, ref_indices),
                                    batch_size=64, shuffle=False, num_workers=0)

            # local anchor batch for drift debug (first ref batch)
            local_anchor_batch = next(iter(loader_ref))[0]

            dbg, prom, sig = clients[cid].train_one_round(
                train_loader=loader_train,
                local_anchor_batch=local_anchor_batch,
                reference_loader_for_signature=loader_ref,
                allow_med_to_slow=(not freeze_slow_now)
            )

            losses.append(dbg["loss"])
            local_drifts.append(dbg["localDrift"])
            sigs.append(sig)

            promo["f2m"] += int(prom["fast→med"])
            promo["m2s"] += int(prom["med→slow"])
            promo["blocked"] += int(prom["blocked"])

        # aggregate med every round
        g_med = fedavg([clients[c].get_med() for c in sel])

        # aggregate slow/LoRA only if (a) period hits and (b) not frozen
        did_slow_agg = 0
        if (r + 1) % SLOW_PERIOD == 0 and (not freeze_slow_now):
            g_slow = fedavg([clients[c].get_slow() for c in sel])
            g_lora = fedavg([clients[c].get_lora() for c in sel])
            did_slow_agg = 1

        # disagreement after training
        d_srv = server_disagreement_from_signatures(sigs)
        freeze_slow_next = (d_srv > DRIFT_TH_SERVER)

        # global eval (stable)
        evaluator = NestedClient(ClientCfg(999), ViTBackboneWithSlowLoRA())
        evaluator.set_med(g_med)
        evaluator.set_slow(g_slow)
        evaluator.set_lora(g_lora)
        accA = evaluator.eval_acc(loaderA, use_fast=False)
        accB = evaluator.eval_acc(loaderB, use_fast=False)

        # debug: detect gate transitions
        trans = ""
        if last_freeze is None:
            last_freeze = freeze_slow_now
        if freeze_slow_now != last_freeze:
            trans = f"  [GATE-TRANSITION now={int(freeze_slow_now)}]"
        last_freeze = freeze_slow_now

        tag = ""
        if r > 0 and schedule[r - 1] == "B" and schedule[r] == "A":
            tag = "  <== RECOVERY POINT (B->A)"

        print(
            f"[R{r:02d} | global={reg_global} | sel={len(sel)} | slow_agg={did_slow_agg} | "
            f"freeze_now={int(freeze_slow_now)} -> freeze_next={int(freeze_slow_next)}]{trans} "
            f"clientRegs={''.join(regs_c)} | "
            f"loss(mean)={np.mean(losses):.3f} | "
            f"localDrift(mean)={np.mean(local_drifts):.3f} | "
            f"serverDisagree={d_srv:.4f} | "
            f"promo(f→m={promo['f2m']}, m→s={promo['m2s']}, blocked={promo['blocked']}) | "
            f"GlobalAcc(A={accA:.3f}, B={accB:.3f}){tag}"
        )

        freeze_slow_now = freeze_slow_next

    print("\nDone.\n")


if __name__ == "__main__":
    main()
