"""
2_model.py
==========
DrugCNN — 4-Block Convolutional Neural Network for 100-class drug classification.

아키텍처:
    입력 (B, 3, 128, 128)
    → Conv Block 1: Conv(3→32)   / BN / ReLU / MaxPool(2×2) → (B, 32,  64, 64)
    → Conv Block 2: Conv(32→64)  / BN / ReLU / MaxPool(2×2) → (B, 64,  32, 32)
    → Conv Block 3: Conv(64→128) / BN / ReLU / MaxPool(2×2) → (B, 128, 16, 16)
    → Conv Block 4: Conv(128→256)/ BN / ReLU / MaxPool(2×2) → (B, 256,  8,  8)
    → Flatten → (B, 16384)
    → FC1(16384→512) / ReLU / Dropout(p=0.5)
    → FC2(512→100)
    → Softmax (추론 시), CrossEntropyLoss 내부 포함 (학습 시)

파라미터 수: ≈ 8.83M
"""

import torch
import torch.nn as nn


# ──────────────────── 기본 컨볼루션 블록 ─────────────────────────

class ConvBlock(nn.Module):
    """
    Conv2d(kernel=3, padding=1) → BatchNorm2d → ReLU → MaxPool2d(2×2)

    padding=1 설정으로 Conv 통과 후 H, W 크기 유지.
    MaxPool(2×2)에서 절반으로 감소.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,     # BN 뒤에 bias는 불필요 (BN의 β가 대체)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ──────────────────────── DrugCNN ────────────────────────────────

class DrugCNN(nn.Module):
    """
    Args:
        num_classes (int): 분류 클래스 수. 기본값 100.
        dropout_p   (float): Dropout 탈락 확률. 기본값 0.5.
    """

    def __init__(self, num_classes: int = 100, dropout_p: float = 0.5):
        super().__init__()

        # ── Feature Extractor ──
        self.features = nn.Sequential(
            ConvBlock(3,   32),   # (B,  3, 128,128) → (B,  32, 64, 64)
            ConvBlock(32,  64),   # (B, 32,  64, 64) → (B,  64, 32, 32)
            ConvBlock(64,  128),  # (B, 64,  32, 32) → (B, 128, 16, 16)
            ConvBlock(128, 256),  # (B,128,  16, 16) → (B, 256,  8,  8)
        )

        # ── Classifier ──
        # Flatten 후 크기: 256 × 8 × 8 = 16,384
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)            # (B, 256, 8, 8)
        x = x.view(x.size(0), -1)      # (B, 16384)
        x = self.classifier(x)          # (B, num_classes)
        return x                        # 로짓 반환 (Softmax는 추론 시 별도 적용)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """추론 시 Softmax 확률 반환"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """추론 시 최고 확률 클래스 인덱스 반환"""
        return self.predict_proba(x).argmax(dim=1)


# ──────────────────── 파라미터 수 계산 유틸 ──────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (1, 3, 128, 128)):
    """레이어별 출력 크기와 파라미터 수 출력"""
    print("=" * 60)
    print(f"  DrugCNN 모델 요약")
    print("=" * 60)

    x = torch.zeros(input_size)
    print(f"  입력:           {tuple(x.shape)}")

    for i, block in enumerate(model.features):
        x = block(x)
        params = sum(p.numel() for p in block.parameters())
        print(f"  Conv Block {i+1}:   {tuple(x.shape)}  |  params: {params:,}")

    x = x.view(x.size(0), -1)
    print(f"  Flatten:        {tuple(x.shape)}")

    for layer in model.classifier:
        x = layer(x)
        if hasattr(layer, 'weight'):
            params = sum(p.numel() for p in layer.parameters())
            print(f"  {layer.__class__.__name__:<16} {tuple(x.shape)}  |  params: {params:,}")

    total = count_parameters(model)
    print("=" * 60)
    print(f"  총 학습 파라미터: {total:,}  ({total/1e6:.2f}M)")
    print("=" * 60)


# ────────────────────────── 실행 (확인용) ────────────────────────

if __name__ == "__main__":
    model = DrugCNN(num_classes=100, dropout_p=0.5)
    print_model_summary(model)

    # 순전파 테스트
    dummy = torch.randn(4, 3, 128, 128)  # batch=4
    model.eval()
    with torch.no_grad():
        logits = model(dummy)
        probs  = model.predict_proba(dummy)
        preds  = model.predict(dummy)

    print(f"\n  순전파 테스트:")
    print(f"  로짓  shape : {logits.shape}")   # (4, 100)
    print(f"  확률  shape : {probs.shape}")    # (4, 100)
    print(f"  예측  shape : {preds.shape}")    # (4,)
    print(f"  확률 합계  : {probs.sum(dim=1)}")  # 각 배치 ≈ 1.0
