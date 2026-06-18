"""
imbalance.py
============
클래스 불균형(현재 최다/최소 = 7.0배) 대응 도구 모음.
세 가지 방법을 제공하며, 3_train.py 에서 플래그로 선택한다.

  1) weighted  : CrossEntropyLoss 에 클래스 빈도 역수 가중치 부여 (기본값)
  2) focal     : Focal Loss — 쉬운(다수) 샘플의 손실을 낮춰 어려운(소수) 샘플에 집중
  3) sampler   : WeightedRandomSampler — 소수 클래스 샘플을 더 자주 추출(oversampling)

세 방법 모두 일반적인 학습 기법으로, 서로 조합해 쓸 수도 있다.
"""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────── 클래스별 가중치 (inverse frequency) ────────────────

def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    클래스 빈도의 역수를 정규화해 가중치 벡터를 만든다.
        w_c = N / (num_classes * n_c)
    n_c 가 작을수록(소수 클래스) 가중치가 커져 손실에 더 크게 반영된다.
    데이터에 없는 클래스는 가중치 1.0 으로 둔다.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.ones(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        n_c = counts.get(c, 0)
        if n_c > 0:
            weights[c] = total / (num_classes * n_c)
    return weights


# ──────────────── Oversampling 용 샘플 가중치 ────────────────

def make_sample_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """
    WeightedRandomSampler 에 넣을 '샘플별' 가중치.
    각 샘플 가중치 = 1 / (그 샘플이 속한 클래스의 빈도).
    → 소수 클래스 샘플이 뽑힐 확률이 높아져 배치 내 클래스가 균형을 이룬다.
    """
    counts = Counter(labels)
    return torch.tensor(
        [1.0 / counts[c] for c in labels], dtype=torch.float32
    )


# ──────────────────────── Focal Loss ────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma: 잘 맞춘(쉬운) 샘플의 손실 기여를 (1-p_t)^gamma 로 줄여,
             아직 못 맞춘(어려운) 소수 클래스에 학습을 집중시킨다. gamma=0 이면 일반 CE.
    - alpha: 클래스별 가중치(weighted CE 의 가중치와 동일한 역할). 선택적.
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # log p_t : 정답 클래스의 로그확률
        log_prob = F.log_softmax(logits, dim=1)
        log_pt = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        focal = (1.0 - pt) ** self.gamma * (-log_pt)

        if self.alpha is not None:
            at = self.alpha.to(logits.device).gather(0, target)
            focal = at * focal

        return focal.mean()


# ──────────────────────── 손실/샘플러 빌더 ────────────────────────

def build_criterion(
    method: str,
    labels: list[int],
    num_classes: int,
    gamma: float,
    device: torch.device,
) -> nn.Module:
    """
    method ∈ {"ce", "weighted", "focal"}
      - ce       : 일반 CrossEntropyLoss (불균형 미보정, 비교 기준선)
      - weighted : 클래스 가중 CrossEntropyLoss (기본 권장)
      - focal    : Focal Loss (+ 클래스 가중치를 alpha 로 사용)
    """
    if method == "ce":
        return nn.CrossEntropyLoss()

    weights = compute_class_weights(labels, num_classes).to(device)

    if method == "weighted":
        return nn.CrossEntropyLoss(weight=weights)
    if method == "focal":
        return FocalLoss(gamma=gamma, alpha=weights)

    raise ValueError(f"알 수 없는 loss method: {method}")
