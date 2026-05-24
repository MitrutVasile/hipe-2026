#!/usr/bin/env python3
"""
HIPE-2026 Model with FOL: XLM-RoBERTa + dual heads + KG fusion + FOL features
                          + Logic-Constrained Loss
=============================================================================
Architecture:
  XLM-R encoder
       │
       ▼ (CLS token, [B, hidden])
  Concat with KG features [B, 16] and FOL features [B, 6]
       │
       ▼ [B, hidden + 16 + 6]
  Shared dropout
       │
       ├─► at_head    Linear → 3 logits  (FALSE, PROBABLE, TRUE)
       └─► isAt_head  Linear → 2 logits  (FALSE, TRUE)

Loss = CE(at) + CE(isAt) + λ_hard * L_hard_constraints + λ_soft * L_soft_constraints

Hard constraints (from logic):
  C1: at = FALSE → isAt = FALSE
        violation prob = P(at=FALSE) * P(isAt=TRUE)
  C2 (data-aware): kg.dead → isAt = FALSE
        violation = kg_dead * P(isAt=TRUE)
  C3 (data-aware): kg.unborn → at = FALSE AND isAt = FALSE
        violation = kg_unborn * (P(at!=FALSE) + P(isAt=TRUE))

Soft constraints (from FOL features detected in text):
  S1: action_flag → at != FALSE
  S2: role_flag and not action_flag → at = PROBABLE (or TRUE only marginally)
  S3: candidacy_flag → at = PROBABLE
  S4: departure_flag → isAt = FALSE  (left = not currently there)
  S5: temporal_now and action → isAt = TRUE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig


class FocalLoss(nn.Module):
    """Focal loss for multi-class with optional class weights."""
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        if self.alpha is not None and self.alpha.device != logits.device:
            self.alpha = self.alpha.to(logits.device)
        ce = F.cross_entropy(logits, targets, weight=self.alpha,
                             reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        mask = (targets != self.ignore_index).float()
        if mask.sum() > 0:
            return (focal * mask).sum() / mask.sum()
        return focal.mean()


class HipeModelFOL(nn.Module):
    """
    Hipe model with First-Order Logic constraint loss.

    KG features (16): from Wikidata facts
    FOL features (6): action, role, origin, temporal, departure, candidacy
    """
    def __init__(self, encoder_name="FacebookAI/xlm-roberta-large",
                 num_at=3, num_isAt=2,
                 kg_dim=16, fol_dim=6,
                 dropout=0.2,
                 use_kg=True, use_fol=True,
                 at_class_weights=None,
                 isAt_class_weights=None,
                 use_focal_at=True,
                 focal_gamma=2.0,
                 lambda_hard=0.3,
                 lambda_soft=0.1):
        super().__init__()
        self.use_kg = use_kg
        self.use_fol = use_fol
        self.lambda_hard = lambda_hard
        self.lambda_soft = lambda_soft

        self.encoder = AutoModel.from_pretrained(encoder_name)
        cfg = AutoConfig.from_pretrained(encoder_name)
        self.hidden_size = cfg.hidden_size

        fusion_dim = self.hidden_size
        if use_kg: fusion_dim += kg_dim
        if use_fol: fusion_dim += fol_dim

        head_hidden = self.hidden_size // 2

        self.dropout = nn.Dropout(dropout)
        self.at_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_at),
        )
        self.isAt_head = nn.Sequential(
            nn.Linear(fusion_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_isAt),
        )

        if use_focal_at:
            self.at_loss = FocalLoss(gamma=focal_gamma, alpha=at_class_weights, ignore_index=-1)
        else:
            self.at_loss = nn.CrossEntropyLoss(weight=at_class_weights, ignore_index=-1)
        self.isAt_loss = nn.CrossEntropyLoss(weight=isAt_class_weights, ignore_index=-1)

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        return cls

    def compute_constraint_loss(self, at_probs, isAt_probs, kg_features, fol_features):
        """
        Compute logic-constrained loss.

        Args:
            at_probs:    [B, 3] softmax output (FALSE=0, PROBABLE=1, TRUE=2)
            isAt_probs:  [B, 2] softmax output (FALSE=0, TRUE=1)
            kg_features: [B, kg_dim] — kg[:, 2]=dead, kg[:, 3]=unborn
            fol_features:[B, fol_dim] — [action, role, origin, temporal, departure, candidacy]

        Returns: dict with 'hard' and 'soft' losses (scalars).
        """
        eps = 1e-8
        p_at_false = at_probs[:, 0]      # P(at=FALSE)
        p_at_prob  = at_probs[:, 1]      # P(at=PROBABLE)
        p_at_true  = at_probs[:, 2]      # P(at=TRUE)
        p_isAt_true = isAt_probs[:, 1]   # P(isAt=TRUE)
        # p_isAt_false = 1 - p_isAt_true

        # ============================================================
        # HARD constraints (from logic / KG facts)
        # ============================================================
        # C1: at=FALSE → isAt=FALSE
        # Violation probability: P(at=FALSE) * P(isAt=TRUE) (both events being TRUE = violation)
        viol_C1 = (p_at_false * p_isAt_true).mean()

        # C2: KG.dead → isAt=FALSE
        if kg_features is not None and kg_features.size(1) > 2:
            dead_flag = kg_features[:, 2]  # 1.0 if person dead at publication
            viol_C2 = (dead_flag * p_isAt_true).mean()
        else:
            viol_C2 = torch.tensor(0.0, device=at_probs.device)

        # C3: KG.unborn → at=FALSE AND isAt=FALSE
        if kg_features is not None and kg_features.size(1) > 3:
            unborn_flag = kg_features[:, 3]
            # Violation: unborn AND (at != FALSE OR isAt = TRUE)
            # = unborn * (P(at != FALSE) + P(isAt = TRUE))
            viol_C3 = (unborn_flag * ((1 - p_at_false) + p_isAt_true)).mean()
        else:
            viol_C3 = torch.tensor(0.0, device=at_probs.device)

        L_hard = viol_C1 + viol_C2 + viol_C3

        # ============================================================
        # SOFT constraints (from FOL text patterns)
        # ============================================================
        if fol_features is not None and fol_features.size(1) >= 6:
            action_flag = fol_features[:, 0]
            role_flag = fol_features[:, 1]
            # origin_flag = fol_features[:, 2]  # informative but not constraint-worthy
            temporal_flag = fol_features[:, 3]
            departure_flag = fol_features[:, 4]
            candidacy_flag = fol_features[:, 5]

            # S1: action verb → at != FALSE  (i.e. push P(at=FALSE) down)
            viol_S1 = (action_flag * p_at_false).mean()

            # S2: role marker (without action) → at != TRUE (push toward PROBABLE)
            # When role and not action, penalize P(at=TRUE)
            role_only = role_flag * (1 - action_flag)
            viol_S2 = (role_only * p_at_true).mean()

            # S3: candidacy → at = PROBABLE (penalize TRUE and FALSE)
            viol_S3 = (candidacy_flag * (p_at_true + p_at_false)).mean()

            # S4: departure → isAt = FALSE (penalize TRUE)
            viol_S4 = (departure_flag * p_isAt_true).mean()

            # S5: temporal NOW + action → isAt = TRUE (penalize FALSE)
            now_action = temporal_flag * action_flag
            viol_S5 = (now_action * (1 - p_isAt_true)).mean()

            L_soft = viol_S1 + viol_S2 + viol_S3 + viol_S4 + viol_S5
        else:
            L_soft = torch.tensor(0.0, device=at_probs.device)

        return {"hard": L_hard, "soft": L_soft, "C1": viol_C1, "C2": viol_C2, "C3": viol_C3}

    def forward(self, input_ids, attention_mask,
                kg_features=None, fol_features=None,
                at_labels=None, isAt_labels=None,
                return_constraints=False):
        cls = self.encode(input_ids, attention_mask)

        # Fuse KG + FOL features
        feats = [cls]
        if self.use_kg and kg_features is not None:
            feats.append(kg_features)
        if self.use_fol and fol_features is not None:
            feats.append(fol_features)
        fused = torch.cat(feats, dim=-1) if len(feats) > 1 else cls
        fused = self.dropout(fused)

        at_logits = self.at_head(fused)
        isAt_logits = self.isAt_head(fused)

        loss = None
        loss_breakdown = {}
        if at_labels is not None and isAt_labels is not None:
            l_at = self.at_loss(at_logits, at_labels)
            l_is = self.isAt_loss(isAt_logits, isAt_labels)
            loss_ce = l_at + l_is

            # Constraint loss
            at_probs = F.softmax(at_logits, dim=-1)
            isAt_probs = F.softmax(isAt_logits, dim=-1)
            constr = self.compute_constraint_loss(at_probs, isAt_probs, kg_features, fol_features)
            loss = loss_ce + self.lambda_hard * constr["hard"] + self.lambda_soft * constr["soft"]

            loss_breakdown = {
                "ce_at": l_at.detach(),
                "ce_isAt": l_is.detach(),
                "hard": constr["hard"].detach(),
                "soft": constr["soft"].detach(),
                "total": loss.detach(),
            }

        out = {
            "loss": loss,
            "at_logits": at_logits,
            "isAt_logits": isAt_logits,
        }
        if return_constraints:
            out["loss_breakdown"] = loss_breakdown
        return out

    @torch.no_grad()
    def predict(self, input_ids, attention_mask,
                kg_features=None, fol_features=None,
                apply_hard_rules=True):
        """Returns dict with at_preds (0/1/2), isAt_preds (0/1), and probs."""
        out = self.forward(input_ids, attention_mask, kg_features, fol_features)
        at_probs = F.softmax(out["at_logits"], dim=-1)
        isAt_probs = F.softmax(out["isAt_logits"], dim=-1)
        at_preds = at_probs.argmax(dim=-1)
        isAt_preds = isAt_probs.argmax(dim=-1)

        if apply_hard_rules and kg_features is not None and kg_features.size(1) >= 4:
            dead = kg_features[:, 2] > 0.5
            unborn = kg_features[:, 3] > 0.5
            isAt_preds = torch.where(dead | unborn, torch.zeros_like(isAt_preds), isAt_preds)
            at_preds = torch.where(unborn, torch.zeros_like(at_preds), at_preds)

        return {
            "at_preds": at_preds,
            "isAt_preds": isAt_preds,
            "at_probs": at_probs,
            "isAt_probs": isAt_probs,
        }


# Keep alias for backwards compat
HipeModel = HipeModelFOL


if __name__ == "__main__":
    print("Testing HipeModelFOL...")
    model = HipeModelFOL(use_kg=True, use_fol=True, lambda_hard=0.3, lambda_soft=0.1)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    B, T = 4, 32
    input_ids = torch.randint(0, 50000, (B, T))
    attention_mask = torch.ones(B, T)
    kg = torch.rand(B, 16)
    fol = torch.rand(B, 6)
    at_labels = torch.tensor([0, 1, 2, -1])
    isAt_labels = torch.tensor([0, 1, 0, -1])

    out = model(input_ids, attention_mask, kg, fol, at_labels, isAt_labels, return_constraints=True)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Breakdown: {out['loss_breakdown']}")
    print(f"at_logits: {out['at_logits'].shape}, isAt_logits: {out['isAt_logits'].shape}")
    print("OK")
