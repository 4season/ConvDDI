"""
5_evaluate.py
=============
학습이 끝난 모델을 '독립 test split' 으로만 평가하는 최종 평가 스크립트.
(train/val 은 절대 사용하지 않는다 → 일반화 성능을 정직하게 측정)

산출 지표
---------
  - Top-1 / Top-5 Accuracy
  - Macro F1-score (클래스별 F1 의 단순 평균 → 불균형에 둔감한 지표)
  - Weighted F1-score (참고용)
  - 클래스별 Precision / Recall / F1 표 (classification report)
  - Confusion Matrix (100×100) → PNG + CSV 저장

출력물
------
  output/eval/metrics.json                요약 지표
  output/eval/classification_report.csv   클래스별 P/R/F1
  output/eval/confusion_matrix.csv        혼동행렬 (행=정답, 열=예측)
  output/eval/confusion_matrix.png        혼동행렬 히트맵

선행 단계
---------
  python src/0_split.py        # splits.json
  python src/3_train.py        # best_model.pth

실행
----
  python src/5_evaluate.py
  python src/5_evaluate.py --checkpoint output/checkpoints/best_model.pth --batch 256
"""

import argparse
import importlib.util as _ilu
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CROPPED_DIR  = PROJECT_ROOT / "data" / "cropped"
SPLITS_PATH  = PROJECT_ROOT / "data" / "splits.json"
CLASS_MAP    = PROJECT_ROOT / "data" / "class_map.json"
DEFAULT_CKPT = PROJECT_ROOT / "output" / "checkpoints" / "best_model.pth"
EVAL_DIR     = PROJECT_ROOT / "output" / "eval"


def _load(mod_name: str, file_name: str):
    spec = _ilu.spec_from_file_location(mod_name, SCRIPT_DIR / file_name)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DrugCNN = _load("model_2", "2_model.py").DrugCNN


class TestDataset(Dataset):
    def __init__(self, items, cropped_dir, num_classes, transform):
        self.transform = transform
        self.samples = [
            (cropped_dir / it["path"], it["class_idx"])
            for it in items if it["class_idx"] < num_classes
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def load_class_names(num_classes: int) -> list[str]:
    raw = json.load(open(CLASS_MAP, encoding="utf-8"))
    idx2name = {info["class_idx"]: info["name"] for _, info in raw.items()}
    return [idx2name.get(i, f"class_{i}") for i in range(num_classes)]


def confusion_matrix_np(y_true, y_pred, n):
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: np.ndarray):
    """혼동행렬에서 클래스별 precision/recall/f1/support 계산."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall    = np.divide(tp, support,  out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                   out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


@torch.no_grad()
def run_inference(model, loader, device, k=5):
    """전체 test셋 추론 → (정답, top1예측, top1맞음여부, topk맞음여부) 수집."""
    model.eval()
    y_true, y_pred = [], []
    top1_correct, topk_correct, total = 0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        topk = logits.topk(k, dim=1).indices.cpu()
        pred1 = topk[:, 0]
        for i, lbl in enumerate(labels):
            y_true.append(int(lbl)); y_pred.append(int(pred1[i]))
            if lbl == pred1[i]:
                top1_correct += 1
            if lbl in topk[i]:
                topk_correct += 1
            total += 1
    return (np.array(y_true), np.array(y_pred),
            top1_correct / total, topk_correct / total)


def save_confusion_png(cm: np.ndarray, save_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[평가] matplotlib 미설치 — 혼동행렬 PNG 스킵")
        return

    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float),
                     where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix (row-normalized, test split)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[평가] 혼동행렬 PNG 저장: {save_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[평가] 디바이스: {device}")

    splits = json.load(open(SPLITS_PATH, encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes = ckpt.get("num_classes", args.classes)
    print(f"[평가] 모델 로드: {args.checkpoint} "
          f"(epoch={ckpt.get('epoch')}, val_acc={ckpt.get('val_acc', 0):.4f})")

    if "splits_seed" in ckpt and ckpt["splits_seed"] != splits["meta"]["seed"]:
        print(f"  ⚠ 경고: 모델 학습 분할 seed({ckpt['splits_seed']}) ≠ "
              f"현재 splits.json seed({splits['meta']['seed']}). 동일 분할로 평가하세요.")

    model = DrugCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_set = TestDataset(splits["test"], CROPPED_DIR, num_classes, get_eval_transform())
    test_loader = DataLoader(test_set, batch_size=args.batch, shuffle=False,
                             num_workers=(4 if device.type == "cuda" else 0))
    print(f"[평가] test 이미지: {len(test_set):,}장 | 클래스 {num_classes}개\n")

    y_true, y_pred, top1, top5 = run_inference(model, test_loader, device, k=5)

    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    precision, recall, f1, support = per_class_prf(cm)
    macro_f1 = float(f1.mean())
    weighted_f1 = float(np.average(f1, weights=support)) if support.sum() else 0.0

    print("=" * 55)
    print("  Test 평가 결과 (독립 분할)")
    print("=" * 55)
    print(f"  Top-1 Accuracy : {top1*100:6.2f}%")
    print(f"  Top-5 Accuracy : {top5*100:6.2f}%")
    print(f"  Macro F1       : {macro_f1:6.4f}")
    print(f"  Weighted F1    : {weighted_f1:6.4f}")
    print("=" * 55)

    names = load_class_names(num_classes)

    order = np.argsort(f1)
    print("\n  F1 최저 5개 클래스:")
    for i in order[:5]:
        print(f"    [{i:3d}] {names[i][:28]:<28} "
              f"P={precision[i]:.2f} R={recall[i]:.2f} F1={f1[i]:.2f} (n={int(support[i])})")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({
        "top1_accuracy": top1, "top5_accuracy": top5,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "num_classes": num_classes, "n_test": int(support.sum()),
        "splits_seed": splits["meta"]["seed"],
    }, open(EVAL_DIR / "metrics.json", "w", encoding="utf-8"),
        ensure_ascii=False, indent=2)

    with open(EVAL_DIR / "classification_report.csv", "w", encoding="utf-8-sig") as f:
        f.write("class_idx,name,precision,recall,f1,support\n")
        for i in range(num_classes):
            f.write(f"{i},{names[i]},{precision[i]:.4f},"
                    f"{recall[i]:.4f},{f1[i]:.4f},{int(support[i])}\n")

    np.savetxt(EVAL_DIR / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_confusion_png(cm, EVAL_DIR / "confusion_matrix.png")

    print(f"\n[평가] 저장 완료 → {EVAL_DIR}")


def parse_args():
    p = argparse.ArgumentParser(description="DrugCNN test-set evaluation")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--classes", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
