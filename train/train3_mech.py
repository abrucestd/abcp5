#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a 130-feature duo score regressor for Train_two_new_mech.cpp output.
The training objective emphasizes high-score accuracy and conservative screening ratios.
"""

import os
import sys
import subprocess
import random
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,Subset
from torch.utils.data import WeightedRandomSampler

wandb = None

IN_DIM        = 130
ROLE_DIM      = 65
DROPOUT_P     = 0.15
LR            = 4e-4
WEIGHT_DECAY  = 1e-5
STEP_SIZE     = 8
GAMMA         = 0.3
BATCH_SIZE    = 2048
EPOCHS        = 125
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 20
PRETRAIN_EPOCHS = 3
SEED          = 42
WANDB_PROJECT = "namerena-duo-regressor"
DEFAULT_DATA_FILES = ["data_mech.csv"]
HARD_CASE_CSV = "actual_ge4650_min_error50_new.csv"
HARD_CASE_ERROR_MIN = -150.0
HARD_CASE_WEIGHT_MULTIPLIER = 2.0
EXPERT_KEEP_WEIGHT = 0.025
EXPERT_GATE_SPARSITY = 0.002
# --------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_test', dest='split_test', action='store_true', help='split a held-out test set')
    parser.add_argument('--no_split_test', dest='split_test', action='store_false', help='do not split a held-out test set')
    parser.set_defaults(split_test=True)
    parser.add_argument('--csv', nargs='+', default=DEFAULT_DATA_FILES, help='Train_two_new.cpp output csv file(s)')
    parser.add_argument('--name', default='reg43d_mech_v0.1', help='output model/figure prefix')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='held-out test ratio')
    parser.add_argument(
        '--wandb',
        choices=['disabled', 'offline', 'online'],
        default='disabled',
        help='wandb logging mode; disabled by default to avoid upload hangs',
    )
    return parser.parse_args()


def _load_hard_case_indices(csv_path):
    """Return source-row indices from the hard-case report for one input CSV."""
    candidates = [HARD_CASE_CSV]
    repo_root_report = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        HARD_CASE_CSV,
    )
    if repo_root_report not in candidates:
        candidates.append(repo_root_report)
    report_path = next((path for path in candidates if os.path.exists(path)), None)
    if report_path is None:
        return set()
    source_name = os.path.basename(os.path.normpath(csv_path)).lower()
    result = set()
    try:
        with open(report_path, encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if float(row.get('error', 'nan')) < HARD_CASE_ERROR_MIN:
                    continue
                report_source = os.path.basename(row.get('source_file', '')).lower()
                if report_source != source_name:
                    continue
                # global_index is zero-based; source_row is one-based.
                result.add(int(row['global_index']))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f'[WARN] failed to load hard-case weights from {HARD_CASE_CSV}: {exc}')
        return set()
    return result

class CSVDataset(Dataset):
    def __init__(self, csv_paths):
        data=[]
        self.special_weight_indices = []
        row_offset = 0
        for p in csv_paths:
            try:
                raw = np.loadtxt(p, delimiter=',', dtype=np.float32)
            except Exception as e:
                print(f"[WARN] failed to load {p}: {e}")
                continue

            if raw.size == 0:
                print(f"[WARN] empty data file skipped: {p}")
                continue

            if raw.ndim == 1:
                raw = raw.reshape(1, -1)

            if raw.shape[1] < IN_DIM + 1:
                print(f"[WARN] column count too small (need >= {IN_DIM + 1}), skipped: {p}, shape={raw.shape}")
                continue

            data.append(raw)

            # The hard-case report stores zero-based global_index values into
            # the source CSV.  Keep those rows marked so the main sampler can
            # give them an additional high-class-like weight without changing
            # the validation/test targets or feature values.
            hard_indices = _load_hard_case_indices(p)
            if hard_indices:
                local = np.asarray(sorted(hard_indices), dtype=np.int64)
                local = local[(local >= 0) & (local < raw.shape[0])]
                self.special_weight_indices.extend((row_offset + local).tolist())
            row_offset += raw.shape[0]

        if len(data) == 0:
            raise ValueError("No usable training csv files.")

        data=np.concatenate(data,axis=0)
        self.special_weight_indices = np.asarray(self.special_weight_indices, dtype=np.int64)

        # ================= y =================
        y_raw = data[:, 0]
        self.y_bias = float(y_raw.mean())
        self.y_scale = float(y_raw.std() + 1e-6)
        y = (y_raw - self.y_bias) / self.y_scale

        # ================= x =================
        x = data[:, 1:IN_DIM + 1]    # shape (N, 130)

        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0) + 1e-6
        self.x_mean = torch.from_numpy(x_mean).float()
        self.x_std = torch.from_numpy(x_std).float()
        x = (x - x_mean) / x_std

        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

        # debug / eval
        self.y_min = float(y_raw.min())
        self.y_max = float(y_raw.max())

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def recover_y(self, y_pred):
        return y_pred * self.y_scale + self.y_bias
class IllusionNet(nn.Module):
    def __init__(self, in_dim=10, out_dim=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.Mish(),
            nn.Dropout(0.10),
            nn.Linear(64, out_dim),
            nn.Mish(),
        )

    def forward(self, x):
        return self.net(x)


class SkillNet(nn.Module):
    def __init__(self, in_dim=48, direct_dim=160, cross_dim=80):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 288),
            nn.LayerNorm(288),
            nn.Mish(),
            nn.Dropout(0.12),
            nn.Linear(288, 288),
            nn.Mish(),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(288, direct_dim),
            nn.Mish(),
        )
        self.cross_head = nn.Sequential(
            nn.Linear(288, cross_dim),
            nn.Mish(),
        )

    def forward(self, x):
        h = self.trunk(x)
        return self.direct_head(h), self.cross_head(h)


class MechanismNet(nn.Module):
    def __init__(self, clone_tail_dim=32, summon_dim=18, state_dim=36, out_dim=24):
        super().__init__()
        self.clone_tail_net = nn.Sequential(
            nn.Linear(clone_tail_dim, 40),
            nn.LayerNorm(40),
            nn.Mish(),
            nn.Dropout(0.20),
            nn.Linear(40, 32),
            nn.Mish(),
        )
        self.summon_net = nn.Sequential(
            nn.Linear(summon_dim, 24),
            nn.LayerNorm(24),
            nn.Mish(),
            nn.Dropout(0.12),
            nn.Linear(24, 20),
            nn.Mish(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 36),
            nn.LayerNorm(36),
            nn.Mish(),
            nn.Dropout(0.20),
            nn.Linear(36, 28),
            nn.Mish(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(80, 48),
            nn.LayerNorm(48),
            nn.Mish(),
            nn.Dropout(0.18),
            nn.Linear(48, out_dim),
            nn.Mish(),
        )

    def forward(self, clone_tail, summon_shadow, state_skill):
        clone_h = self.clone_tail_net(clone_tail)
        summon_h = self.summon_net(summon_shadow)
        state_h = self.state_net(state_skill)
        return self.fuse(torch.cat([clone_h, summon_h, state_h], dim=1))


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.fc(x)


class LifeWheelExpert(nn.Module):
    """A bounded residual expert for life-wheel self-state interactions.

    The duo contains two allies, not an explicit enemy.  Consequently this
    branch models the life-wheel holder's own HP/attributes and difficult
    same-role skill structures first, with only a small auxiliary projection
    for teammate skill synergy.  It never treats teammate HP as a swap target.
    """

    FEATURE_VERSION = 4

    def __init__(self, in_dim=58, hidden=64, residual_cap=0.75):
        super().__init__()
        self.register_buffer('feature_version', torch.tensor(self.FEATURE_VERSION, dtype=torch.uint8))
        self.legacy_activation = False
        self.residual_cap = float(residual_cap)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 32),
            nn.Mish(),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Mish(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.residual = nn.Linear(32, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features, return_gate=False):
        hidden = self.trunk(features)
        gate = self.gate(features).squeeze(-1)
        # life_wheel_features lays out each role as 25 values followed by the
        # second role.  Use the two life-wheel strengths explicitly; the old
        # implementation accidentally compared role 1 life with role 1 low HP
        # and never activated for role-2-only life-wheel cases.
        if self.legacy_activation:
            # Keep v0.5 checkpoints numerically reproducible when they are
            # inspected; newly trained v0.6 models use the corrected branch.
            life_activation = torch.sigmoid(
                4.0 * (torch.maximum(features[:, 0], features[:, 1]) - 1.0)
            )
        else:
            life1 = features[:, 0]
            life2 = features[:, 25]
            life_activation = torch.sigmoid(4.0 * (torch.maximum(life1, life2) - 1.0))
        # Keep the expert bounded in normalized-score space.  With the usual
        # score scale this is a finite correction of roughly a few hundred
        # score points at most, while the initial correction is exactly zero.
        output = self.residual_cap * life_activation * gate * torch.tanh(self.residual(hidden).squeeze(-1))
        if return_gate:
            return output, gate, life_activation
        return output


class PairSynergyExpert(nn.Module):
    """A bounded residual branch for rare cross-role interactions.

    The input is built from shared, order-symmetric role features.  HP is
    intentionally excluded here: life-wheel self-state handling belongs to
    LifeWheelExpert and must not learn a teammate-HP swap proxy.
    """

    FEATURE_VERSION = 1

    def __init__(self, in_dim=192, hidden=96, residual_cap=1.0):
        super().__init__()
        self.register_buffer('feature_version', torch.tensor(self.FEATURE_VERSION, dtype=torch.uint8))
        self.residual_cap = float(residual_cap)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.Mish(),
            nn.Dropout(0.10),
            nn.Linear(hidden, 48),
            nn.Mish(),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.Mish(),
            nn.Linear(48, 1),
            nn.Sigmoid(),
        )
        self.residual = nn.Linear(48, 1)
        # Preserve the base model exactly until this expert is trained.
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.constant_(self.gate[-2].bias, -3.5)

    def forward(self, features, return_gate=False):
        hidden = self.trunk(features)
        gate = self.gate(features).squeeze(-1)
        output = self.residual_cap * gate * torch.tanh(self.residual(hidden).squeeze(-1))
        if return_gate:
            return output, gate
        return output


def load_state_dict_compatible(model, state_dict):
    """Load checkpoints while isolating expert feature-version changes."""
    state_dict = dict(state_dict)
    incoming_life = state_dict.get('life_expert.feature_version')
    incoming_pair = state_dict.get('pair_expert.feature_version')
    legacy_life = incoming_life is not None and int(incoming_life) == 3
    if incoming_life is None or (
        int(incoming_life) != LifeWheelExpert.FEATURE_VERSION and not legacy_life
    ):
        state_dict = {key: value for key, value in state_dict.items()
                      if not key.startswith('life_expert.')}
    elif legacy_life:
        # v0.5 has the same 58-dimensional parameter shapes.  Keep its
        # learned weights and select the old activation only for compatibility
        # inspection; new training always uses FEATURE_VERSION 4.
        state_dict.pop('life_expert.feature_version', None)
    if incoming_pair is None or int(incoming_pair) != PairSynergyExpert.FEATURE_VERSION:
        state_dict = {key: value for key, value in state_dict.items()
                      if not key.startswith('pair_expert.')}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_prefixes = ('life_expert.', 'pair_expert.')
    allowed_missing = [key for key in missing if key.startswith(allowed_prefixes)]
    unexpected = list(unexpected)
    unexpected_allowed = [key for key in unexpected if key.startswith(allowed_prefixes)]
    unexpected = [key for key in unexpected if key not in unexpected_allowed]
    missing = [key for key in missing if key not in allowed_missing]
    if missing or unexpected:
        raise RuntimeError(
            f'incompatible checkpoint: missing={missing[:8]}, unexpected={unexpected[:8]}'
        )
    if allowed_missing:
        print(f'[INFO] initialized {len(allowed_missing)} versioned expert tensors')
    if legacy_life:
        model.life_expert.legacy_activation = True
        print('[INFO] loaded legacy v0.5 life-expert activation for reproducibility')
    return model


class Regressor43D(nn.Module):
    def __init__(self):
        super().__init__()

        self.illusion_net = IllusionNet(in_dim=10, out_dim=12)
        self.skill_net = SkillNet(in_dim=48, direct_dim=160, cross_dim=80)
        self.mechanism_net = MechanismNet(clone_tail_dim=32, summon_dim=18, state_dim=36, out_dim=24)
        self.main_in = nn.Sequential(
            nn.Linear(464, 1024),
            nn.LayerNorm(1024),
            nn.Mish()
        )
        
        self.res_blocks = nn.Sequential(
            ResBlock(1024, dropout=0.15),
            ResBlock(1024, dropout=0.15),
            ResBlock(1024, dropout=0.12),
            nn.Linear(1024, 384),
            nn.LayerNorm(384),
            nn.Mish(),
        )
        self.base_head = nn.Linear(384, 1)
        self.tail_head = nn.Sequential(
            nn.Linear(384, 128),
            nn.Mish(),
            nn.Linear(128, 1)
        )
        self.tail_gate = nn.Sequential(
            nn.Linear(384, 64),
            nn.Mish(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.life_expert = LifeWheelExpert(in_dim=58, hidden=64, residual_cap=0.75)
        self.pair_expert = PairSynergyExpert(in_dim=192, hidden=96, residual_cap=1.0)

    @staticmethod
    def life_wheel_features(role1, role2):
        """Build self-state and limited teammate-skill interaction features.

        Inputs are standardized model features.  The explicit triple terms
        are the main addition: life-wheel with the same role's low HP, clone,
        amulet, and shield structure.  Illusion is intentionally absent.
        """
        def build_self(role):
            life = F.softplus(role[:, 17:18])
            low_hp = F.softplus(-role[:, 0:1])
            attrs = role[:, 0:8]
            skill_indices = (31, 33, 34, 35, 36, 37, 39, 41, 42)
            skills = [F.softplus(role[:, index:index + 1]) for index in skill_indices]

            def interaction(*values):
                product = values[0]
                for value in values[1:]:
                    product = product * value
                return torch.log1p(product)

            clone, shield, amulet = skills[0], skills[5], skills[4]
            return torch.cat([
                life,
                low_hp,
                interaction(life, low_hp),
                life * attrs,
                *[interaction(life, skill) for skill in skills],
                interaction(life, low_hp, clone),
                interaction(life, low_hp, amulet),
                interaction(life, clone, amulet),
                interaction(life, low_hp, clone, amulet),
                interaction(life, clone, shield),
            ], dim=1), life, skills

        self1, life1, skills1 = build_self(role1)
        self2, life2, skills2 = build_self(role2)
        # Secondary cross-role terms use skills only; no teammate HP is used.
        teammate_skill_indices = (0, 4, 5, 7)  # clone, amulet, shield, dying
        cross = [
            torch.log1p(life1 * skills2[index]) for index in teammate_skill_indices
        ] + [
            torch.log1p(life2 * skills1[index]) for index in teammate_skill_indices
        ]
        return torch.cat([self1, self2, *cross], dim=1)

    @staticmethod
    def pair_synergy_features(role1, role2):
        """Return shared-role, order-symmetric interaction features.

        HP (local feature 0) is excluded so this generic branch cannot learn a
        life-wheel teammate-HP exchange shortcut.  The remaining 64 features
        cover attributes, skills, phantom values, and tail mechanism fields.
        ``tanh`` keeps products bounded after position-wise standardization.
        """
        role1 = torch.tanh(role1[:, 1:ROLE_DIM])
        role2 = torch.tanh(role2[:, 1:ROLE_DIM])
        return torch.cat([
            role1 + role2,
            torch.abs(role1 - role2),
            role1 * role2,
        ], dim=1)

    def forward(self, x, return_components=False):
        """
        x: (B, 130), Train_two_new_mech.cpp layout: role1(65) + role2(65).
        """

        role1 = x[:, :ROLE_DIM]
        role2 = x[:, ROLE_DIM:ROLE_DIM * 2]

        # f32 is clone/split skill; f45:f54 are summon/clone attributes in Train_two_new.cpp.
        illu_feat1 = torch.cat([role1[:, 32:33], role1[:, 45:54]], dim=1)
        illu_feat2 = torch.cat([role2[:, 32:33], role2[:, 45:54]], dim=1)

        skill_feat1 = torch.cat([role1[:, 8:32], role1[:, 33:45]], dim=1)
        skill_feat2 = torch.cat([role2[:, 8:32], role2[:, 33:45]], dim=1)
        main_feat = torch.cat([role1[:, :8], role2[:, :8]], dim=1)

        illu_strength1 = self.illusion_net(illu_feat1)
        illu_strength2 = self.illusion_net(illu_feat2)
        skill_feat1 = torch.cat([skill_feat1, illu_strength1], dim=1)
        skill_feat2 = torch.cat([skill_feat2, illu_strength2], dim=1)
        skl_direct1, skl_cross1 = self.skill_net(skill_feat1)
        skl_direct2, skl_cross2 = self.skill_net(skill_feat2)
        cross_skill = skl_cross1 * skl_cross2

        # f54:f64 encode clone-gated name_base tail-active and tail-seat rules.
        # These features are needed for clone reload / decay-floor effects that are not recoverable
        # from the final skill strengths alone.
        clone_tail = torch.cat([
            role1[:, 17:18],
            role1[:, 23:25],
            role1[:, 31:33],
            role2[:, 17:18],
            role2[:, 23:25],
            role2[:, 31:33],
            role1[:, 54:65],
            role2[:, 54:65],
        ], dim=1)
        summon_shadow = torch.cat([role1[:, 45:54], role2[:, 45:54]], dim=1)
        state_skill = torch.cat([
            role1[:, 23:33],
            role2[:, 23:33],
            role1[:, 37:45],
            role2[:, 37:45],
        ], dim=1)
        mechanism = self.mechanism_net(clone_tail, summon_shadow, state_skill)

        feat = torch.cat([
            main_feat,
            skl_direct1,
            skl_direct2,
            cross_skill,
            mechanism,
            illu_strength1,
            illu_strength2,
        ], dim=1)
        out = self.main_in(feat)
        h = self.res_blocks(out)
        base = self.base_head(h)
        tail = self.tail_head(h)
        gate = self.tail_gate(h)
        life_features = self.life_wheel_features(role1, role2)
        life_residual, life_gate, life_activation = self.life_expert(
            life_features, return_gate=True
        )
        pair_features = self.pair_synergy_features(role1, role2)
        pair_residual, pair_gate = self.pair_expert(pair_features, return_gate=True)
        base_output = base.squeeze(-1) + gate.squeeze(-1) * tail.squeeze(-1)
        output = base_output + life_residual + pair_residual
        if return_components:
            return output, base_output, life_residual, pair_residual, life_gate, life_activation, pair_gate
        return output

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    dataset,
):
    """
    Training helper; comments were normalized after an encoding issue.
    """

    model.train()

    bloss_sum,hloss_sum,lloss_sum, mae_sum, rmse_sum = 0.0, 0.0, 0.0,0.0,0.0
    maxerr, minerr = -1e9, 1e9

    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset

    # ========= raw/norm helper =========
    y_bias = getattr(dataset, 'y_bias', 0.0)
    y_scale = float(dataset.y_scale)

    def raw_to_norm(v_raw):
        return (v_raw - y_bias) / y_scale

    P = 3.2
    LAMBDA = 3.8
    LAMBDA2 = 0.62
    LAMBDA3 = 1.30

    # 新评分版本整体约下移 150 分，保持原有 loss 分段语义。
    low_ref = raw_to_norm(4150.0)
    EPS = 1e-6

    def raw_width(v_raw):
        return float(v_raw) / y_scale

    def smooth_score_gate(y_norm, center_raw, width_raw):
        return torch.sigmoid((y_norm - raw_to_norm(center_raw)) / (raw_width(width_raw) + EPS))

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        (
            pred,
            base_pred,
            life_residual,
            pair_residual,
            life_gate,
            life_activation,
            pair_gate,
        ) = model(x, return_components=True)
        pred = pred.view(-1)

        base_point = torch.nn.functional.smooth_l1_loss(pred, y, beta=0.25, reduction='none')

        mid_gate = smooth_score_gate(y, 4350.0, 220.0)
        high_gate = smooth_score_gate(y, 4700.0, 240.0)
        base_w = 0.55 + 0.45 * mid_gate + 5.0 * high_gate.pow(2.0)
        high_under_w = 1.0 + 6.5 * high_gate.pow(2.0)
        base_loss = (base_point * base_w).mean()

        beta_eff = 0.25 / (1.0 + 4.5 * high_gate.pow(P))

        under = torch.relu(y - pred)

        high_loss = torch.where(
            under < beta_eff,
            0.5 * under.pow(2) / beta_eff,
            under - 0.5 * beta_eff
        )

        high_under_loss = (high_loss * high_under_w).mean()

        over_estimation = torch.relu(pred - y)
        high_pred_gate = torch.sigmoid((pred - raw_to_norm(4550.0)) / 0.1)
        low_over_loss = (over_estimation.pow(2) * high_pred_gate).mean()

        low_gate = torch.sigmoid((low_ref - y) / (raw_width(220.0) + EPS))
        low_zone_w = 1.0 + 4.0 * low_gate.pow(2.0)
        low_over_bias_loss = (over_estimation.pow(2) * low_zone_w).mean()

        loss = base_loss + LAMBDA * high_under_loss + LAMBDA2 * low_over_loss + LAMBDA3 * low_over_bias_loss

        # Keep the new experts local while they learn.  Low/mid-score replay
        # should stay close to the base model; high-score cases get a softer
        # penalty so rare synergistic corrections can emerge.
        expert_delta = life_residual + pair_residual
        ordinary_weight = 1.0 - high_gate.detach()
        expert_keep_loss = (expert_delta.pow(2) * (0.25 + ordinary_weight)).mean()
        gate_sparse_loss = (
            (ordinary_weight * pair_gate).mean()
            + ((1.0 - life_activation.detach()) * life_gate).mean()
        )
        loss = loss + EXPERT_KEEP_WEIGHT * expert_keep_loss + EXPERT_GATE_SPARSITY * gate_sparse_loss

        loss = loss + 2e-5 * sum(p.abs().sum() for p in model.parameters())

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_raw = pred * y_scale + y_bias
            y_raw    = y    * y_scale + y_bias

            err = pred_raw - y_raw
            abs_err = err.abs()

            bloss_sum += base_loss.item() * x.size(0)
            hloss_sum += LAMBDA*high_under_loss.item() * x.size(0)
            lloss_sum += (LAMBDA2 * low_over_loss.item() + LAMBDA3 * low_over_bias_loss.item()) * x.size(0)
            mae_sum  += abs_err.sum().item()
            rmse_sum += ((pred - y) ** 2).sum().item()

            maxerr = max(maxerr, err.max().item())
            minerr = min(minerr, err.min().item())

    n = len(loader.dataset)
    return (
        bloss_sum / n,
        hloss_sum/n,
        lloss_sum/n,
        mae_sum  / n,
        np.sqrt(rmse_sum / n),
        maxerr,
        minerr,
    )

@torch.no_grad()
def validate(model, loader,  device, dataset, detailed_stats=False):
    model.eval()

    preds_all, trues_all = [], []
    loss_sum, mae_sum, rmse_sum = 0., 0., 0.
    maxerr, minerr = -1e9, 1e9
    max_id, min_id, offset = 0, 0, 0
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    y_bias = getattr(dataset, 'y_bias', 0.0)
    y_scale = float(dataset.y_scale)

    for x, y in loader:
        batch_size = x.size(0)
        x = x.to(device)
        y = y.to(device).view(-1)

        pred = model(x).view(-1)

        pred_raw = (pred * y_scale + y_bias).cpu().numpy()
        y_raw    = (y    * y_scale + y_bias).cpu().numpy()

        err = pred_raw - y_raw
        abs_err = np.abs(err)

        if detailed_stats:
            preds_all.append(pred_raw)
            trues_all.append(y_raw)
        loss=torch.nn.functional.smooth_l1_loss(
            pred, y, beta=0.3
        )
        loss_sum += loss.item() * x.size(0)
        mae_sum  += abs_err.sum()
        rmse_sum += ((pred - y) ** 2).sum().item()

        if err.max() > maxerr:
            maxerr = err.max()
            max_id = offset + err.argmax()

        if err.min() < minerr:
            minerr = err.min()
            min_id = offset + err.argmin()

        offset += batch_size

    ratio4650, ratio4750 = np.nan, np.nan
    if detailed_stats:
        preds_all = np.concatenate(preds_all)
        trues_all = np.concatenate(trues_all)

        def safe_screen_ratio(threshold):
            mask = trues_all >= threshold
            actual = np.count_nonzero(mask)
            if actual == 0:
                return 0.0, np.nan, 0
            safe_min = np.min(preds_all[mask])
            selected = np.count_nonzero(preds_all >= safe_min)
            return selected / actual, safe_min, actual

        for high_th in [4650, 4750]:
            high_mask = trues_all >= high_th
            if np.any(high_mask):
                high_err = preds_all[high_mask] - trues_all[high_mask]
                ratio, safe_min, actual = safe_screen_ratio(high_th)
                print(
                    f"High>={high_th}: N={actual}, MAE={np.mean(np.abs(high_err)):.2f}, "
                    f"MinErr={np.min(high_err):.1f}, SafePredMin={safe_min:.1f}, Ratio={ratio:.2f}x"
                )
            else:
                print(f"High>={high_th}: N=0")

        low_mask = trues_all <= 4150
        if np.any(low_mask):
            low_err = preds_all[low_mask] - trues_all[low_mask]
            print("Low-score MeanErr:", np.mean(low_err))
            print("Low-score OverRate:", np.mean(low_err > 0))

        ratio4650, _, _ = safe_screen_ratio(4650)
        ratio4750, _, _ = safe_screen_ratio(4750)

    n = len(loader.dataset)
    return (
        loss_sum / n,
        mae_sum  / n,
        np.sqrt(rmse_sum / n),
        maxerr,
        minerr,
        max_id,
        min_id,
        ratio4650,
        ratio4750,
    )

def split_train_val(dataset, val_ratio=0.1, seed=42):
    n = len(dataset)
    indices = np.arange(n)

    rng = np.random.default_rng(seed)
    val_size = int(n * val_ratio)

    val_indices = rng.choice(indices, size=val_size, replace=False)
    train_indices = np.setdiff1d(indices, val_indices)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices)
    )


def split_train_val_test(dataset, test_ratio=0.1, val_ratio=0.1, seed=42):
    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)

    test_size = int(n * test_ratio)
    test_indices = rng.choice(indices, size=test_size, replace=False)
    remain_indices = np.setdiff1d(indices, test_indices)

    val_size = int(len(remain_indices) * val_ratio)
    val_indices = rng.choice(remain_indices, size=val_size, replace=False)
    train_indices = np.setdiff1d(remain_indices, val_indices)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices)
    )


def dump_subset_to_csv(subset, out_path):
    if isinstance(subset, torch.utils.data.Subset):
        base = subset.dataset
        indices = np.asarray(subset.indices)
    else:
        base = subset
        indices = np.arange(len(base))

    x_norm = base.x[indices].cpu().numpy()
    y_norm = base.y[indices].cpu().numpy()

    x_mean = base.x_mean.cpu().numpy()
    x_std = base.x_std.cpu().numpy()
    x_raw = x_norm * x_std + x_mean

    y_bias = getattr(base, 'y_bias', 0.0)
    y_raw = y_norm * base.y_scale + y_bias

    output = np.concatenate([y_raw.reshape(-1, 1), x_raw], axis=1)
    np.savetxt(out_path, output, delimiter=',', fmt='%.6f')
    print(f'[INFO] Test split exported: {out_path}, rows={len(output)}')


def ensure_data_diy_ready():
    print("[INFO] Old Train_two_diy data generation is disabled for Train_two_new training.")
    return False

import torch
import torch.nn as nn

class ExtremeAwareSmoothL1(nn.Module):
    def __init__(self, beta=0.2, alpha=1.0, scale=400.0,
                 y_min=0.0, y_max=0.0,
                 w_under=-114.0,
                 w_over=200.0,
                 gamma=0.1):
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.scale = scale
        self.y_min = y_min
        self.y_max = y_max
        self.base = nn.SmoothL1Loss(reduction='none', beta=beta)

        self.w_under = w_under
        self.w_over = w_over
        self.gamma = gamma

    def forward(self, pred_norm, target_norm):
        base_loss = self.base(pred_norm, target_norm)

        delta_norm = pred_norm - target_norm
        err_raw = delta_norm * (self.y_max - self.y_min + 1e-8)
        barrier = torch.relu(err_raw.abs() - self.scale) ** 2

        return base_loss.mean() + 0.06 * barrier.mean() 

class DualExtremeAsymmetricLoss(nn.Module):
    def __init__(
        self,
        base='smoothl1',
        beta=0.2,
        high_th=0.78,
        lambda_under=0.4,
        lambda_over=0.4,
    ):
        super().__init__()

        if base == 'mse':
            self.base_loss = nn.MSELoss()
        else:
            self.base_loss = nn.SmoothL1Loss(beta=beta)

        self.high_th = high_th
        self.lambda_under = lambda_under
        self.lambda_over = lambda_over

    def smooth_gate(self, x, lo=-0.1, hi=0.05):
        z = (x - lo) / (hi - lo)
        return torch.sigmoid(6 * (z - 0.5))

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        loss = self.base_loss(y_pred, y_true)

        tau = 0.03
        under_gate = torch.nn.functional.softplus(y_true - self.high_th, beta=1/tau) * self.smooth_gate(y_true - y_pred) * (1/tau)
        under_penalty = torch.relu(y_true - y_pred) * under_gate

        over_gate = torch.nn.functional.softplus(y_pred - self.high_th, beta=1/tau) * self.smooth_gate(y_pred - y_true) * (1/tau)
        over_penalty = torch.relu(y_pred - y_true) * over_gate

        # KMean
        k = 4

        extreme_under = torch.topk(under_penalty,k).values
        extreme_over = torch.topk(over_penalty, k).values
        loss = (
            loss
            + self.lambda_under * extreme_under.mean()
            + self.lambda_over * extreme_over.mean()
        )
        return loss


def compute_sample_errors(model, dataset, device):
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    errs = []

    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
    else:
        base_dataset = dataset

    y_bias = getattr(base_dataset, 'y_bias', 0.0)
    y_scale = float(base_dataset.y_scale)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).view(-1)        # (B,)

            pred = model(x).view(-1)         # (B,)

            # raw space
            pred_raw = (pred * y_scale + y_bias).cpu().numpy()  # (B,)
            y_raw    = (y    * y_scale + y_bias).cpu().numpy()  # (B,)

            err = np.abs(pred_raw - y_raw)   # (B,)
            errs.append(err)

    return np.concatenate(errs, axis=0)

def build_dynamic_sampler(
    dataset,
    low_th1=3600.0,
    low_th2=4200.0,
    high_start=4500.0,
    high_end=5300.0,
    low_keep1=1,
    low_keep2=1,
    low_keep3=1
):
    """
    Training helper; comments were normalized after an encoding issue.
    """

    if isinstance(dataset, torch.utils.data.Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices)
    else:
        base = dataset
        indices = np.arange(len(dataset))

    # -------- y (raw) --------
    y_norm = base.y[indices]
    y_bias = getattr(base, 'y_bias', 0.0)
    y_raw = y_norm.numpy() * base.y_scale + y_bias

    weights = np.ones_like(y_raw, dtype=np.float64)

    low_mask1 = y_raw < low_th1
    print(f'low_mask1 size:{np.count_nonzero(low_mask1)}')
    weights[low_mask1] = low_keep1
    low_mask2 = (y_raw < low_th2) & (y_raw>=low_th1)
    print(f'low_mask2 size:{np.count_nonzero(low_mask2)}')
    low_mask3 = (y_raw < low_th2+200) & (y_raw>=low_th2)
    print(f'low_mask3 size:{np.count_nonzero(low_mask3)}')
    print(f'common size:{np.count_nonzero((y_raw < high_start) & (y_raw>=low_th2+200))}')
    weights[low_mask2] = low_keep2

    high_mask = y_raw >= high_start
    print(f'high_mask size:{np.count_nonzero(high_mask)}')
    smooth_mask = y_raw >= low_th2
    smooth_y = y_raw[smooth_mask]
    bridge_gate = 1.0 / (1.0 + np.exp(-np.clip((smooth_y - (low_th2 + 200.0)) / 40.0, -60.0, 60.0)))
    bridge_w = low_keep3 + (1.0 - low_keep3) * bridge_gate

    high_t = (smooth_y - high_start) / (high_end - high_start + 1e-6)
    high_t = np.clip(high_t, 0.0, 1.0)
    high_gate = 1.0 / (1.0 + np.exp(-np.clip((smooth_y - (high_start + 20.0)) / 35.0, -60.0, 60.0)))
    high_boost = (3.0 + 18.0 * (high_t ** 2)) * high_gate

    weights[smooth_mask] = bridge_w + high_boost

    special_mask = np.zeros(len(indices), dtype=bool)
    special_global = getattr(base, 'special_weight_indices', np.empty(0, dtype=np.int64))
    if len(special_global):
        special_mask = np.isin(indices, special_global)
        # These are high-score cases whose measured error is still at least
        # -150.  Apply an additional multiplier on top of the normal
        # high-score schedule, matching the existing high-class emphasis.
        weights[special_mask] *= HARD_CASE_WEIGHT_MULTIPLIER
    print(f'hard_case_mask size:{np.count_nonzero(special_mask)}')

    total_w = weights.sum() + 1e-12
    print(f'weight_share low1: {weights[low_mask1].sum()/total_w:.4f}')
    print(f'weight_share low2: {weights[low_mask2].sum()/total_w:.4f}')
    print(f'weight_share low3: {weights[low_mask3].sum()/total_w:.4f}')
    common_mask = (y_raw < high_start) & (y_raw >= low_th2 + 200)
    print(f'weight_share common: {weights[common_mask].sum()/total_w:.4f}')
    print(f'weight_share high: {weights[high_mask].sum()/total_w:.4f}')
    print(f'weight_share hard_case: {weights[special_mask].sum()/total_w:.4f}')

    weights = torch.from_numpy(weights)

    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True
    )

    return sampler
def main():
    global wandb
    args = get_args()
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    name = args.name

    wandb_run = None
    if args.wandb == 'disabled':
        os.environ['WANDB_MODE'] = 'disabled'
        os.environ.setdefault('WANDB_SILENT', 'true')
        wandb = None
        print('[INFO] wandb disabled. Use --wandb offline or --wandb online to enable logging.')
    else:
        os.environ['WANDB_MODE'] = args.wandb
        os.environ.setdefault('WANDB_SILENT', 'true')
        try:
            import wandb as wandb_module
            wandb = wandb_module
            wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=name,
                mode=args.wandb,
                config={
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "seed": SEED,
                    "split_test": args.split_test,
                    "test_ratio": args.test_ratio,
                    "wandb": args.wandb,
                }
            )
        except ImportError:
            wandb = None
            print("[WARN] wandb is not installed; skip wandb logging.")
        except Exception as e:
            print(f"[WARN] wandb init failed, skip online logging: {e}")
            print("[INFO] To enable wandb, run: wandb login")
            wandb_run = None

    # Train_two_new.cpp data is required; do not auto-generate old Train_two_diy data here.
    print(f"[INFO] training csv files: {args.csv}")
    full_dataset = CSVDataset(args.csv)
    print(f'[INFO] hard-case rows (error >= {HARD_CASE_ERROR_MIN:g}): '
          f'{len(full_dataset.special_weight_indices)}')
    if args.split_test:
        train_set, val_set, test_set = split_train_val_test(full_dataset, test_ratio=args.test_ratio, val_ratio=0.1)
        print(f'[INFO] split_test=ON, test_ratio={args.test_ratio}')
    else:
        train_set, val_set = split_train_val(full_dataset, val_ratio=0.1)
        test_set = None
        print('[INFO] split_test=OFF, no held-out test set will be exported.')
    sampler = build_dynamic_sampler(train_set)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,sampler=sampler,   num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    model = Regressor43D().to(device)
    criterion = ExtremeAwareSmoothL1(beta=0.2,alpha=0.6,y_min=full_dataset.y_min,y_max=full_dataset.y_max)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    lambda_under=0.0
    global_step = 0

    def log_epoch_to_wandb(stage_name, epoch_idx, tr_bloss, tr_hloss, tr_lloss, tr_mae, tr_rmse, tr_maxe, tr_mine,
                           val_loss, val_mae, val_rmse, val_maxe, val_mine):
        nonlocal global_step, wandb_run
        if wandb_run is None:
            return
        try:
            wandb.log({
                "stage": stage_name,
                "epoch": epoch_idx,
                "train/base_loss": tr_bloss,
                "train/high_under_loss": tr_hloss,
                "train/low_over_loss": tr_lloss,
                "train/mae": tr_mae,
                "train/rmse": tr_rmse,
                "train/max_err": tr_maxe,
                "train/min_err": tr_mine,
                "val/loss": val_loss,
                "val/mae": val_mae,
                "val/rmse": val_rmse,
                "val/max_err": val_maxe,
                "val/min_err": val_mine,
            }, step=global_step)
            global_step += 1
        except Exception as e:
            print(f"[WARN] wandb.log failed, disable wandb: {e}")
            wandb_run = None

    hist = {k: [] for k in ['train_loss', 'val_loss', 'train_mae', 'val_mae', 'train_rmse', 'val_rmse', 'train_max', 'val_max', 'train_min', 'val_min']}
    def should_report_detail(epoch_idx):
        return epoch_idx % 5 == 0

    def ratio_summary(ratio4650, ratio4750):
        if np.isfinite(ratio4650) and np.isfinite(ratio4750):
            return f'{ratio4650:3f}x+{ratio4750:3f}'
        return 'ratio skipped'

    for epoch in range(-PRETRAIN_EPOCHS, 0): # sampler 1
        tr_bloss,tr_hloss,tr_lloss, tr_mae, tr_rmse, tr_maxe,tr_mine = train_one_epoch(model, train_loader,  optimizer, device,full_dataset)
        val_loss, val_mae, val_rmse, val_maxe,val_mine,max_id,min_id,a,b = validate(
            model, val_loader, device, full_dataset, detailed_stats=should_report_detail(epoch)
        )

        scheduler.step()

        for k, v in zip(hist.keys(), [tr_bloss+tr_hloss, val_loss, tr_mae, val_mae, tr_rmse, val_rmse, tr_maxe, val_maxe,tr_mine,val_mine]):
            hist[k].append(v)

        print(f'pretrain Epoch {epoch:3d}/{EPOCHS} | '
                f'train base L={tr_bloss:.8f} high_under L={tr_hloss:.8f} low_up L={tr_lloss:.8f} MAE={tr_mae:.3f} RMSE={tr_rmse:.5f} MaxErr={tr_maxe:.2f} MinErr={tr_mine:.2f}| '
                f'val L={val_loss:.8f} MAE={val_mae:.3f} RMSE={val_rmse:.5f} MaxErr={val_maxe:.2f} MinErr={val_mine:.2f}|')
        log_epoch_to_wandb("pretrain", epoch, tr_bloss, tr_hloss, tr_lloss, tr_mae, tr_rmse, tr_maxe, tr_mine,
                   val_loss, val_mae, val_rmse, val_maxe, val_mine)
    for p in model.main_in.parameters():
        p.requires_grad = False

    for epoch in range(1, STAGE1_EPOCHS+1): # sampler 1
        tr_bloss,tr_hloss,tr_lloss,  tr_mae, tr_rmse, tr_maxe,tr_mine = train_one_epoch(model, train_loader,  optimizer, device,full_dataset)
        val_loss, val_mae, val_rmse, val_maxe,val_mine,max_id,min_id,a,b = validate(
            model, val_loader, device, full_dataset, detailed_stats=should_report_detail(epoch)
        )

        scheduler.step()

        for k, v in zip(hist.keys(), [tr_bloss+tr_hloss, val_loss, tr_mae, val_mae, tr_rmse, val_rmse, tr_maxe, val_maxe,tr_mine,val_mine]):
            hist[k].append(v)

        print(f'Stage 1 Epoch {epoch:3d}/{EPOCHS} | {ratio_summary(a, b)} '
                f'train base L={tr_bloss:.8f} high_under L={tr_hloss:.8f} low_up L={tr_lloss:.8f} MAE={tr_mae:.3f} RMSE={tr_rmse:.5f} MaxErr={tr_maxe:.2f} MinErr={tr_mine:.2f}| '
                f'val L={val_loss:.8f} MAE={val_mae:.3f} RMSE={val_rmse:.5f} MaxErr={val_maxe:.2f} MinErr={val_mine:.2f}|')
        log_epoch_to_wandb("stage1", epoch, tr_bloss, tr_hloss, tr_lloss, tr_mae, tr_rmse, tr_maxe, tr_mine,
                   val_loss, val_mae, val_rmse, val_maxe, val_mine)
    for p in model.main_in.parameters():
        p.requires_grad = True

    errs = compute_sample_errors(model, train_set, device).squeeze()
    weights = 1.0 + (errs / np.percentile(errs, 90))**2
    weights = np.clip(weights, 1.0, 50.0)
    if len(full_dataset.special_weight_indices):
        train_indices = np.asarray(train_set.indices if isinstance(train_set, Subset)
                                   else np.arange(len(train_set)), dtype=np.int64)
        hard_mask = np.isin(train_indices, full_dataset.special_weight_indices)
        weights[hard_mask] *= HARD_CASE_WEIGHT_MULTIPLIER
        print(f'[INFO] stage2 hard-case rows weighted: {np.count_nonzero(hard_mask)}')
    weights = torch.from_numpy(weights).double().squeeze()
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler,  num_workers=2, pin_memory=torch.cuda.is_available())


    
    for epoch in range(1, STAGE2_EPOCHS+1): # sampler 1
        display_epoch = epoch + STAGE1_EPOCHS
        tr_bloss,tr_hloss,tr_lloss, tr_mae, tr_rmse, tr_maxe,tr_mine = train_one_epoch(model, train_loader,  optimizer, device,full_dataset)
        val_loss, val_mae, val_rmse, val_maxe,val_mine,max_id,min_id,a,b = validate(
            model, val_loader, device, full_dataset, detailed_stats=should_report_detail(display_epoch)
        )

        scheduler.step()

        for k, v in zip(hist.keys(), [tr_bloss+tr_hloss, val_loss, tr_mae, val_mae, tr_rmse, val_rmse, tr_maxe, val_maxe,tr_mine,val_mine]):
            hist[k].append(v)

        print(f'Stage 2 Epoch {display_epoch:3d}/{EPOCHS} | {ratio_summary(a, b)} '
                f'train base L={tr_bloss:.8f} high_under L={tr_hloss:.8f} low_up L={tr_lloss:.8f} MAE={tr_mae:.3f} RMSE={tr_rmse:.5f} MaxErr={tr_maxe:.2f} MinErr={tr_mine:.2f}| '
                f'val L={val_loss:.8f} MAE={val_mae:.3f} RMSE={val_rmse:.5f} MaxErr={val_maxe:.2f} MinErr={val_mine:.2f}|')
        log_epoch_to_wandb("stage2", epoch + STAGE1_EPOCHS, tr_bloss, tr_hloss, tr_lloss, tr_mae, tr_rmse, tr_maxe, tr_mine,
                   val_loss, val_mae, val_rmse, val_maxe, val_mine)


    # sampler back to random
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=torch.cuda.is_available())
    # criterion change
    # criterion = DualExtremeAsymmetricLoss(
    #     base='smoothl1',
    #     beta=0.2,
    #     high_th=0.78,
    #     lambda_under=0.3,
    #     lambda_over=0.4,
    # )
    
    for epoch in range(STAGE1_EPOCHS+STAGE2_EPOCHS+1,EPOCHS): # sampler 1
        lambda_under=min(0.25,0.03*epoch)
        tr_bloss,tr_hloss,tr_lloss, tr_mae, tr_rmse, tr_maxe,tr_mine = train_one_epoch(model, train_loader,  optimizer, device,full_dataset)
        val_loss, val_mae, val_rmse, val_maxe,val_mine,max_id,min_id,a,b = validate(
            model, val_loader, device, full_dataset, detailed_stats=should_report_detail(epoch)
        )

        scheduler.step()

        for k, v in zip(hist.keys(), [tr_bloss+tr_hloss, val_loss, tr_mae, val_mae, tr_rmse, val_rmse, tr_maxe, val_maxe,tr_mine,val_mine]):
            hist[k].append(v)

        print(f'Stage 3 Epoch {epoch:3d}/{EPOCHS} | {ratio_summary(a, b)} '
                f'train base L={tr_bloss:.8f} high_under L={tr_hloss:.8f} low_up L={tr_lloss:.8f} MAE={tr_mae:.3f} RMSE={tr_rmse:.5f} MaxErr={tr_maxe:.2f} MinErr={tr_mine:.2f}| '
                f'val L={val_loss:.8f} MAE={val_mae:.3f} RMSE={val_rmse:.5f} MaxErr={val_maxe:.2f} MinErr={val_mine:.2f}|')
        log_epoch_to_wandb("stage3", epoch, tr_bloss, tr_hloss, tr_lloss, tr_mae, tr_rmse, tr_maxe, tr_mine,
                   val_loss, val_mae, val_rmse, val_maxe, val_mine)

   
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(hist['train_loss'], label='train')
    plt.plot(hist['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('SmoothL1'); plt.legend(); plt.title('Loss')

    plt.subplot(1,3,2)
    plt.plot(hist['train_mae'], label='train MAE')
    plt.plot(hist['val_mae'], label='val MAE')
    plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend(); plt.title('MAE')

    plt.subplot(1,3,3)
    plt.plot(hist['train_max'], label='train MaxErr')
    plt.plot(hist['val_max'], label='val MaxErr')
    plt.xlabel('Epoch'); plt.ylabel('Max Abs Error'); plt.legend(); plt.title('MaxErr')

    val_loss, val_mae, val_rmse, val_maxe,val_mine,max_id,min_id,Overall_k,Overall_b = validate(
        model, val_loader, device, full_dataset, detailed_stats=True
    )

    torch.save({
        "model": model.state_dict(),
        "y_scale": full_dataset.y_scale,
        "y_bias": full_dataset.y_bias,
        "x_mean": full_dataset.x_mean,
        "x_std": full_dataset.x_std,
        "architecture": "reg43d_mech_life_pair_expert_v0.6",
        "life_expert": {
            "enabled": True,
            "feature_dim": 58,
            "residual_cap_normalized": 0.75,
            "feature_version": LifeWheelExpert.FEATURE_VERSION,
            "activation": "sigmoid(4*(max(life1,life2)-1))",
            "error_case_threshold": HARD_CASE_ERROR_MIN,
        },
        "pair_expert": {
            "enabled": True,
            "feature_dim": 192,
            "residual_cap_normalized": 1.0,
            "feature_version": PairSynergyExpert.FEATURE_VERSION,
            "features": "tanh(role1[1:])+tanh(role2[1:]), abs(diff), product",
            "hp_excluded": True,
        },
    }, f"{name}.pt")
    print(f'Finished. Model -> {name}.pt')

    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()

    if args.split_test and test_set is not None:
        test_csv = f'{name}_test_split.csv'
        dump_subset_to_csv(test_set, test_csv)

        print('[INFO] Running tester4_new_mech.py on test split...')
        r1 = subprocess.run([
            sys.executable,
            'tester4_new_mech.py',
            '--model', f'{name}.pt',
            '--npy', test_csv,
        ], check=False)
        print(f'[INFO] tester4_new_mech.py exit code: {r1.returncode}')

        print('[INFO] Segment distribution plot skipped; use tester4_new_mech.py summary for Train_two_new_mech models.')
        r2 = None
        # tester4_segment_dist.py imports train.py and is kept for old checkpoints.
        # Create a train3-based copy before using it with Train_two_new checkpoints.
        #
        # subprocess.run([...])
        #
        #
        #
        #
    else:
        print('[INFO] skipped test split export and automatic tester.')

    if wandb_run is not None:
        try:
            wandb.finish(quiet=True)
        except TypeError:
            wandb.finish()
        except Exception as e:
            print(f"[WARN] wandb.finish failed: {e}")

if __name__ == '__main__':
    main()
