"""
1_preprocess.py
===============
COCO bbox 좌표를 이용해 4약물 원본 이미지에서 개별 약물을 크롭하고
128×128 으로 리사이즈한 뒤 학습용 디렉터리 구조로 저장합니다.

출력 구조:
    data/cropped/
        {class_idx:03d}_{drug_code}/   (예: 000_K-031863/)
            {원본파일명}.png

실행:
    python src/1_preprocess.py
    python src/1_preprocess.py --dry-run   # 실제 파일 저장 없이 통계만 확인
"""

import argparse
import json
import glob
import os
from pathlib import Path
from PIL import Image

# ────────────────────────── 경로 설정 ─────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LABEL_BASE   = PROJECT_ROOT / "data" / "labels"
IMAGE_BASE   = PROJECT_ROOT / "data" / "images"
OUTPUT_BASE  = PROJECT_ROOT / "data" / "cropped"
CLASS_MAP_PATH = PROJECT_ROOT / "data" / "class_map.json"

# 이미지 크기
TARGET_SIZE = (128, 128)

# TL/TS 번호 범위
SPLIT_RANGE = range(1, 9)   # 1 ~ 8


# ─────────────────────────── 헬퍼 ─────────────────────────────────

def load_class_map() -> dict:
    """class_map.json 로드 → {drug_code: class_idx}"""
    with open(CLASS_MAP_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return {code: info["class_idx"] for code, info in raw.items()}


def get_output_dir(class_idx: int, drug_code: str) -> Path:
    """출력 폴더: data/cropped/000_K-031863/"""
    folder_name = f"{class_idx:03d}_{drug_code}"
    return OUTPUT_BASE / folder_name


def safe_crop(img: Image.Image, bbox: list) -> Image.Image | None:
    """
    COCO 포맷 bbox [x_min, y_min, w, h] 로 크롭.
    유효하지 않은 bbox는 None 반환.
    """
    x, y, w, h = [int(v) for v in bbox]
    W, H = img.size

    # 경계 보정
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    return img.crop((x1, y1, x2, y2))


# ─────────────────────────── 핵심 로직 ────────────────────────────

def process_single_json(
    json_path: Path,
    class_map: dict,
    ts_num: int,
    dry_run: bool,
    stats: dict,
) -> None:
    """JSON 1개 처리: 크롭 → 리사이즈 → 저장"""
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        stats["json_error"] += 1
        return

    images      = data.get("images", [])
    annotations = data.get("annotations", [])

    if not images or not annotations:
        stats["skip_empty"] += 1
        return

    img_info = images[0]
    ann      = annotations[0]
    drug_code = img_info.get("dl_mapping_code", "")

    # 선정된 100개 클래스에 없으면 스킵
    if drug_code not in class_map:
        stats["skip_not_selected"] += 1
        return

    class_idx = class_map[drug_code]
    file_name = img_info.get("file_name", "")
    bbox      = ann.get("bbox", [])

    if not file_name or len(bbox) != 4:
        stats["skip_missing_field"] += 1
        return

    # ── 이미지 파일 경로 구성 ──
    # file_name 예: K-000250-000573-002483-006192_0_2_0_2_75_000_200.png
    # combo_code: 파일명에서 첫 번째 '_0_' 이전 부분
    combo_code = file_name.split("_0_")[0] if "_0_" in file_name else file_name.rsplit("_", 1)[0]
    img_path = IMAGE_BASE / f"TS_{ts_num}" / combo_code / file_name

    if not img_path.exists():
        stats["img_not_found"] += 1
        return

    # ── 크롭 & 리사이즈 ──
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        stats["img_open_error"] += 1
        return

    cropped = safe_crop(img, bbox)
    if cropped is None:
        stats["bbox_invalid"] += 1
        return

    resized = cropped.resize(TARGET_SIZE, Image.BILINEAR)

    # ── 저장 ──
    out_dir = get_output_dir(class_idx, drug_code)
    out_path = out_dir / file_name

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        resized.save(out_path, format="PNG")

    stats["success"] += 1


def run_preprocess(dry_run: bool = False) -> None:
    class_map = load_class_map()
    print(f"[전처리] 선정 클래스: {len(class_map)}개")
    print(f"[전처리] 출력 경로: {OUTPUT_BASE}")
    if dry_run:
        print("[전처리] ── DRY-RUN 모드 (파일 저장 없음) ──\n")

    stats = {
        "success":            0,
        "json_error":         0,
        "skip_empty":         0,
        "skip_not_selected":  0,
        "skip_missing_field": 0,
        "img_not_found":      0,
        "img_open_error":     0,
        "bbox_invalid":       0,
    }

    for n in SPLIT_RANGE:
        tl_dir = LABEL_BASE / f"TL_{n}"
        json_files = glob.glob(str(tl_dir / "**" / "*.json"), recursive=True)
        print(f"  TL_{n}: {len(json_files):,}개 JSON 처리 중...", end=" ", flush=True)

        for jp in json_files:
            process_single_json(Path(jp), class_map, n, dry_run, stats)

        print(f"누적 성공: {stats['success']:,}장")

    # ── 결과 요약 ──
    total = sum(stats.values())
    print("\n" + "=" * 55)
    print(f"  전처리 완료 요약")
    print("=" * 55)
    print(f"  성공 저장:          {stats['success']:>8,}장")
    print(f"  클래스 미포함 스킵:  {stats['skip_not_selected']:>8,}장")
    print(f"  이미지 파일 없음:    {stats['img_not_found']:>8,}장")
    print(f"  JSON 파싱 오류:      {stats['json_error']:>8,}장")
    print(f"  bbox 오류:           {stats['bbox_invalid']:>8,}장")
    print(f"  기타:                {total - stats['success'] - stats['skip_not_selected']:>8,}장")
    print("=" * 55)

    if not dry_run and stats["success"] > 0:
        # 클래스별 저장 수 확인
        print("\n클래스별 저장 이미지 수 (상위 10개):")
        class_dirs = sorted(OUTPUT_BASE.iterdir()) if OUTPUT_BASE.exists() else []
        counts = [(d.name, len(list(d.glob("*.png")))) for d in class_dirs if d.is_dir()]
        counts.sort(key=lambda x: -x[1])
        for name, cnt in counts[:10]:
            print(f"  {name:<40} {cnt:,}장")


# ──────────────────────────── 실행 ────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug image preprocessor")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 파일 저장 없이 통계만 출력"
    )
    args = parser.parse_args()
    run_preprocess(dry_run=args.dry_run)
