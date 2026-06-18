"""
6_feature_maps.py
=================
학습된 DrugCNN(best_model.pth)을 불러와, 한 장의 약품 이미지에 대해
각 Conv 블록이 만들어내는 feature map(채널맵)을 시각화한다.

목적:
    - "각 채널이 어떤 패턴에 반응하는가"를 눈으로 확인 (보고서 ⑥의 추정 → 관찰 전환)
    - 얕은 층(저수준 에지·색) vs 깊은 층(고수준 형태·각인) 비교

실행 예:
    python src/6_feature_maps.py \
        --image data/cropped/000_K-031863/<파일명>.png \
        --ckpt  output/best_model.pth \
        --out   output/report_assets/feature_maps

의존성: torch, torchvision, matplotlib, pillow
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# 같은 폴더의 모델 정의 재사용
import importlib.util


# ──────────────── 모델 정의 로드 (2_model.py 재사용) ────────────────

def load_model_def(model_py: Path):
    """2_model.py에서 DrugCNN 클래스를 동적으로 임포트."""
    spec = importlib.util.spec_from_file_location("drug_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DrugCNN


# ──────────────── 전처리 (학습과 동일하게 맞춤) ────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

preprocess = T.Compose([
    T.Resize((128, 128)),                 # 학습 입력 해상도와 동일
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ──────────────── feature map 수집 (forward hook) ─────────────────

def collect_feature_maps(model: nn.Module, x: torch.Tensor):
    """
    각 ConvBlock(model.features[i]) 출력 텐서를 hook으로 가로채 반환.
    반환: [(block_idx, tensor(shape=(1,C,H,W))), ...]
    """
    captured = []

    def make_hook(idx):
        def hook(_module, _inp, out):
            captured.append((idx, out.detach().cpu()))
        return hook

    handles = [blk.register_forward_hook(make_hook(i))
               for i, blk in enumerate(model.features)]

    model.eval()
    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    captured.sort(key=lambda t: t[0])
    return captured


# ──────────────── 시각화 ──────────────────────────────────────────

def plot_block(block_idx: int, fmap: torch.Tensor, out_dir: Path,
               max_channels: int = 16):
    """
    한 블록의 앞쪽 max_channels개 채널을 격자로 저장.
    fmap: (1, C, H, W)
    """
    fmap = fmap[0]                      # (C, H, W)
    n = min(max_channels, fmap.size(0))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = axes.flatten()

    for c in range(n):
        ch = fmap[c]
        # 채널별 min-max 정규화 후 표시 (대비 확보)
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        axes[c].imshow(ch, cmap="viridis")
        axes[c].set_title(f"ch {c}", fontsize=8)
        axes[c].axis("off")
    for c in range(n, len(axes)):
        axes[c].axis("off")

    C, H, W = fmap.shape
    fig.suptitle(f"Conv Block {block_idx + 1}  |  shape (C={C}, {H}x{W})",
                 fontsize=11)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"block{block_idx + 1}_feature_maps.png"
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"  저장: {save_path}")


# ──────────────── 메인 ────────────────────────────────────────────

def main():
    # 실행 위치(cwd)와 무관하게 동작하도록, 기본 경로는 "프로젝트 루트" 기준으로 잡는다.
    # 이 파일이 <root>/src/6_feature_maps.py 이므로 부모의 부모가 프로젝트 루트.
    ROOT = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="시각화할 약품 크롭 이미지 경로 (생략 시 data/cropped에서 자동 선택)")
    ap.add_argument("--data_dir", default=str(ROOT / "data/cropped"),
                    help="--image 생략 시 이미지를 찾을 루트 폴더")
    ap.add_argument("--ckpt", default=str(ROOT / "output/best_model.pth"), help="모델 체크포인트")
    ap.add_argument("--model_py", default=str(ROOT / "src/2_model.py"), help="DrugCNN 정의 파일")
    ap.add_argument("--out", default=str(ROOT / "output/report_assets/feature_maps"), help="출력 폴더")
    ap.add_argument("--num_classes", type=int, default=100)
    ap.add_argument("--max_channels", type=int, default=16, help="블록당 표시 채널 수")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # 1) 모델 로드
    DrugCNN = load_model_def(Path(args.model_py))
    model = DrugCNN(num_classes=args.num_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    # 3_train.py는 {"epoch":.., "model_state": state_dict, ..} 형태로 저장.
    # 순수 state_dict로 저장된 경우도 함께 처리.
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state)
    print(f"[model] 가중치 로드 완료: {args.ckpt}")

    # 2) 이미지 선택 (생략 시 자동)
    image_path = args.image
    if image_path is None:
        candidates = sorted(Path(args.data_dir).rglob("*.png"))
        if not candidates:
            raise SystemExit(f"[오류] {args.data_dir} 에서 .png 이미지를 찾지 못했습니다. "
                             f"--image 로 직접 지정하세요.")
        image_path = str(candidates[0])
        print(f"[auto] --image 미지정 → 자동 선택: {image_path}")

    # 3) 이미지 전처리
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)   # (1, 3, 128, 128)
    print(f"[input] {image_path}  →  tensor {tuple(x.shape)}")

    # 3) feature map 수집 + 저장
    out_dir = Path(args.out)
    captured = collect_feature_maps(model, x)
    for block_idx, fmap in captured:
        plot_block(block_idx, fmap, out_dir, max_channels=args.max_channels)

    # 4) 원본도 함께 저장(비교용)
    out_dir.mkdir(parents=True, exist_ok=True)
    img.resize((128, 128)).save(out_dir / "input_resized.png")
    print(f"  저장: {out_dir / 'input_resized.png'}")

    # 5) 예측 결과 출력(확인용)
    model.eval()
    with torch.no_grad():
        probs = model.predict_proba(x)[0]
        top5 = torch.topk(probs, k=5)
    print("\n[예측 Top-5 (class_idx: prob)]")
    for idx, p in zip(top5.indices.tolist(), top5.values.tolist()):
        print(f"  {idx:>3d}: {p:.4f}")

    print("\n완료. 출력 폴더를 확인하세요:", out_dir)


if __name__ == "__main__":
    main()
