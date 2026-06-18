"""
3_train.py
==========
DrugCNN 학습 파이프라인 (그룹 분할 + 클래스 불균형 대응 버전).

핵심 변경점
-----------
1) 무작위 split 폐지 → data/splits.json 의 'train'/'val' 만 사용한다.
   (combo=사진 단위로 분리된 분할이라 데이터누수가 없음. 0_split.py 참고)
2) test split 은 학습 중 절대 사용하지 않는다. 최종 평가는 5_evaluate.py 전담.
3) 클래스 불균형 대응을 플래그로 선택:
       --loss    ce | weighted | focal     (기본 weighted)
       --sampler none | weighted            (WeightedRandomSampler oversampling)
       --gamma   focal loss 의 집중 계수    (기본 2.0)

선행 단계
---------
    python src/0_split.py          # data/splits.json 생성 (먼저 1회 실행)

실행 예시
---------
    python src/3_train.py
    python src/3_train.py --loss focal --gamma 2.0
    python src/3_train.py --loss weighted --sampler weighted
    python src/3_train.py --classes 20 --epochs 20      # 소규모 실험

결과물
------
    output/checkpoints/best_model.pth   최저 검증 손실 모델 가중치
    output/loss_curve.png               학습/검증 손실·정확도 곡선
"""

import argparse
import importlib.util as _ilu
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

# 파일명이 2_model.py / imbalance.py → importlib 동적 로드
def _load(mod_name: str, file_name: str):
    spec = _ilu.spec_from_file_location(mod_name, Path(__file__).parent / file_name)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_model_mod = _load("model_2", "2_model.py")
_imb       = _load("imbalance", "imbalance.py")
DrugCNN          = _model_mod.DrugCNN
build_criterion  = _imb.build_criterion
make_sample_weights = _imb.make_sample_weights

# ──────────── 경로 ────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CROPPED_DIR  = PROJECT_ROOT / "data" / "cropped"
SPLITS_PATH  = PROJECT_ROOT / "data" / "splits.json"
CLASS_MAP    = PROJECT_ROOT / "data" / "class_map.json"
CKPT_DIR     = PROJECT_ROOT / "output" / "checkpoints"
PLOT_PATH    = PROJECT_ROOT / "output" / "loss_curve.png"

# ──────────── 기본 하이퍼파라미터 ────────────
DEFAULTS = dict(
    lr=1e-3, batch=128, dropout=0.5, epochs=80, patience=8,
    classes=100, seed=42, loss="weighted", sampler="none", gamma=2.0,
)


# ─────────────────────── Dataset ────────────────────────────────

class SplitDataset(Dataset):
    """
    splits.json 의 한 split(list of {"path","class_idx"})을 읽는 Dataset.
    num_classes 로 학습 클래스 수를 제한할 수 있다(소규모 실험용).
    cache=True 시 모든 이미지를 RAM에 올려 I/O 병목을 제거한다.
    """

    def __init__(self, items: list[dict], cropped_dir: Path,
                 transform=None, num_classes: int = 100, cache: bool = False):
        self.transform = transform
        self.samples = [
            (cropped_dir / it["path"], it["class_idx"])
            for it in items if it["class_idx"] < num_classes
        ]
        self.labels = [lbl for _, lbl in self.samples]

        self.cache = cache
        self._img_cache = [None] * len(self.samples)
        if cache:
            for i, (p, _) in enumerate(self.samples):
                self._img_cache[i] = Image.open(p).convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = self._img_cache[idx] if self.cache else \
              Image.open(self.samples[idx][0]).convert("RGB")
        label = self.samples[idx][1]
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────── 데이터 변환 ────────────────────────────────

def get_transforms(is_train: bool) -> transforms.Compose:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    if is_train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ──────────────────── 학습/검증 루프 ─────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total


# ──────────────────── 학습 곡선 저장 ─────────────────────────────

def save_loss_curve(history: dict, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[학습] matplotlib 미설치 — 학습 곡선 저장 스킵")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curve"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   label="Val Acc")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curve"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[학습] 학습 곡선 저장: {save_path}")


# ──────────────────────── 메인 ───────────────────────────────────

def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[학습] 디바이스: {device}")
    if device.type == "cuda":
        print(f"[학습] GPU: {torch.cuda.get_device_name(0)}")

    # ── 분할 로드 ──
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} 가 없습니다. 먼저 `python src/0_split.py` 를 실행하세요."
        )
    splits = json.load(open(SPLITS_PATH, encoding="utf-8"))
    print(f"[학습] 분할 로드: train {len(splits['train']):,} / "
          f"val {len(splits['val']):,} / test {len(splits['test']):,} "
          f"(seed={splits['meta']['seed']})")

    # ── 데이터셋 (train/val 만) ──
    use_cache = (device.type == "cuda")  # GPU면 RAM 캐시로 I/O 제거
    train_set = SplitDataset(splits["train"], CROPPED_DIR,
                             get_transforms(True),  args.classes, cache=use_cache)
    val_set   = SplitDataset(splits["val"],   CROPPED_DIR,
                             get_transforms(False), args.classes, cache=use_cache)
    print(f"[학습] 학습 {len(train_set):,}장 | 검증 {len(val_set):,}장 "
          f"(클래스 {args.classes}개)")

    # ── 불균형 대응: 샘플러 ──
    num_workers = 4 if device.type == "cuda" else 0
    if args.sampler == "weighted":
        w = make_sample_weights(train_set.labels, args.classes)
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler,
                                  num_workers=num_workers,
                                  pin_memory=(device.type == "cuda"))
        print("[학습] oversampling: WeightedRandomSampler 사용")
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers,
                            pin_memory=(device.type == "cuda"))

    # ── 모델 ──
    model = DrugCNN(num_classes=args.classes, dropout_p=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[학습] 파라미터 수: {n_params/1e6:.2f}M")

    # ── 불균형 대응: 손실함수 ──
    criterion = build_criterion(args.loss, train_set.labels, args.classes,
                                args.gamma, device)
    # 검증 손실은 항상 일반 CE 로 측정(early-stopping 기준의 일관성)
    val_criterion = nn.CrossEntropyLoss()
    print(f"[학습] 손실: {args.loss}"
          + (f" (gamma={args.gamma})" if args.loss == "focal" else ""))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Early Stopping ──
    best_val_loss = float("inf")
    patience_counter = 0
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_ckpt = CKPT_DIR / "best_model.pth"
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n[학습] 시작 — lr={args.lr}, batch={args.batch}, dropout={args.dropout}, "
          f"epochs={args.epochs}, patience={args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, val_criterion, device)
        elapsed = time.time() - t0

        for k, v in zip(history, (tr_loss, va_loss, tr_acc, va_acc)):
            history[k].append(v)

        improved = va_loss < best_val_loss
        print(f"  Epoch [{epoch:3d}/{args.epochs}]  "
              f"train_loss: {tr_loss:.4f}  train_acc: {tr_acc:.4f}  "
              f"val_loss: {va_loss:.4f}  val_acc: {va_acc:.4f}  "
              f"({elapsed:.1f}s){' ✓' if improved else ''}")

        if improved:
            best_val_loss = va_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_loss": va_loss, "val_acc": va_acc,
                "num_classes": args.classes, "args": vars(args),
                "splits_seed": splits["meta"]["seed"],
            }, best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[학습] Early stopping (epoch {epoch})")
                break

    print(f"\n[학습] 완료. 최적 모델: {best_ckpt}")
    print("[학습] 최종 성능 평가는 `python src/5_evaluate.py` (test split) 로 수행하세요.")
    save_loss_curve(history, PLOT_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DrugCNN Training (group split + imbalance)")
    p.add_argument("--lr",       type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch",    type=int,   default=DEFAULTS["batch"])
    p.add_argument("--dropout",  type=float, default=DEFAULTS["dropout"])
    p.add_argument("--epochs",   type=int,   default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int,   default=DEFAULTS["patience"])
    p.add_argument("--classes",  type=int,   default=DEFAULTS["classes"],
                   help="학습 클래스 수 (소규모 실험: 10~20)")
    p.add_argument("--seed",     type=int,   default=DEFAULTS["seed"])
    p.add_argument("--loss",     choices=["ce", "weighted", "focal"],
                   default=DEFAULTS["loss"], help="손실 함수 (불균형 대응)")
    p.add_argument("--sampler",  choices=["none", "weighted"],
                   default=DEFAULTS["sampler"], help="oversampling 샘플러")
    p.add_argument("--gamma",    type=float, default=DEFAULTS["gamma"],
                   help="focal loss 집중 계수")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
