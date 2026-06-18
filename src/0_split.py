"""
0_split.py
==========
데이터누수를 막기 위한 '그룹 단위' train/val/test 분할 스크립트.

문제 배경
---------
1) 같은 약품 조합(combo)을 각도·조명만 바꿔 촬영한 near-duplicate 이미지가
   combo 하나당 평균 3장씩 존재한다.
        K-003351-016232-031863_0_2_0_2_70_000_200.png
        K-003351-016232-031863_0_2_0_2_75_000_200.png   ← 같은 장면, 각도만 다름
        K-003351-016232-031863_0_2_0_2_90_000_200.png
2) 하나의 combo(원본 사진) 안에는 서로 다른 약 3~4종이 함께 찍혀 있어,
   그 크롭들이 각각 다른 클래스 폴더로 흩어진다(combo당 평균 3.64개 클래스).
이 두 사실 때문에 이미지를 무작위로 train/val/test에 흩으면, 모델이 '약' 대신
같은 사진이 공유하는 '배경·조명·촬영판'을 외워 검증 정확도가 부풀려지는
데이터누수가 발생한다.

해결
----
파일명에서 '_0_' 앞부분(= combo 식별자, 곧 원본 사진 한 장)을 그룹 키로 삼아,
한 combo의 '모든' 크롭(여러 클래스에 걸친 변형 전부)을 train/val/test 중
오직 한 곳에만 전역(global) 배정한다. 즉 분할 단위는 '사진' 이다.
이렇게 하면 같은 사진의 어떤 조각도 두 split에 동시에 등장하지 않는다.
combo는 전역 무작위로 섞어 배정하되, 100개 클래스가 세 split에 모두
존재하는지(coverage) 검증한다.

출력
----
data/splits.json
    {
      "meta": {...},
      "train": [{"path": "...", "class_idx": 0, "combo": "..."}, ...],
      "val":   [...],
      "test":  [...]
    }

실행
----
    python src/0_split.py                       # 기본 70/15/15
    python src/0_split.py --train 0.7 --val 0.15 --test 0.15 --seed 42
    python src/0_split.py --dry-run             # 통계만 출력, 파일 저장 안 함
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CROPPED_DIR  = PROJECT_ROOT / "data" / "cropped"
SPLITS_PATH  = PROJECT_ROOT / "data" / "splits.json"


def combo_key(file_name: str) -> str:
    """
    파일명에서 촬영 변형 접미사를 떼고 combo 식별자를 만든다.
    'K-003351-016232-031863_0_2_0_2_70_000_200.png'
        → 'K-003351-016232-031863'
    같은 combo의 70/75/90 각도 변형이 하나의 그룹으로 묶인다.
    """
    stem = file_name[:-4] if file_name.endswith(".png") else file_name
    return stem.split("_0_")[0] if "_0_" in stem else stem.rsplit("_", 1)[0]


def collect_class_dirs(cropped_dir: Path) -> list[Path]:
    """data/cropped 아래 '000_K-XXXXXX' 형식의 클래스 폴더만 정렬해 반환."""
    return sorted(
        d for d in cropped_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and d.name.split("_")[0].isdigit()
    )


def gather_images(class_dirs: list[Path]) -> tuple[list[dict], dict[str, set]]:
    """
    모든 크롭 이미지를 수집하고, combo → {그 combo가 포함하는 class_idx 집합} 도 만든다.
    반환:
        items: [{"path": "000_K-031863/xxx.png", "class_idx": 0, "combo": "..."}, ...]
        combo_classes: {combo: {class_idx, ...}}
    """
    items: list[dict] = []
    combo_classes: dict[str, set] = defaultdict(set)
    for cd in class_dirs:
        class_idx = int(cd.name.split("_")[0])
        for img in cd.glob("*.png"):
            c = combo_key(img.name)
            items.append({"path": f"{cd.name}/{img.name}",
                          "class_idx": class_idx, "combo": c})
            combo_classes[c].add(class_idx)
    return items, combo_classes


def assign_combos(
    combo_classes: dict[str, set],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, str]:
    """
    combo 를 전역으로 train/val/test 에 배정한다(combo 하나 = 한 split).
    무작위 배정 후 모든 클래스가 세 split에 존재하지 않으면 시드를 바꿔 재시도.
    """
    n_class = len({ci for s in combo_classes.values() for ci in s})

    for attempt in range(50):
        combos = list(combo_classes.keys())
        rng.shuffle(combos)
        n = len(combos)
        a = int(round(n * ratios[0]))
        b = a + int(round(n * ratios[1]))
        assign = {c: ("train" if i < a else "val" if i < b else "test")
                  for i, c in enumerate(combos)}

        cover = defaultdict(set)
        for c, classes in combo_classes.items():
            for ci in classes:
                cover[ci].add(assign[c])
        ok = all(len(cover[ci]) == 3 for ci in range(n_class) if ci in cover)
        if ok and len(cover) == n_class:
            if attempt:
                print(f"[분할] coverage 보정: {attempt+1}번째 시도에서 전 클래스 포함")
            return assign

    print("[분할] ⚠ 일부 클래스가 특정 split에 없을 수 있습니다.")
    return assign


def run(ratios: tuple[float, float, float], seed: int, dry_run: bool) -> None:
    assert abs(sum(ratios) - 1.0) < 1e-6, "train+val+test 비율 합은 1이어야 합니다."
    rng = random.Random(seed)

    class_dirs = collect_class_dirs(CROPPED_DIR)
    print(f"[분할] 클래스 폴더: {len(class_dirs)}개")
    print(f"[분할] 비율 train/val/test = {ratios[0]:.2f}/{ratios[1]:.2f}/{ratios[2]:.2f}, seed={seed}")

    items, combo_classes = gather_images(class_dirs)
    print(f"[분할] 전역 고유 combo(사진): {len(combo_classes):,}개 | 총 이미지 {len(items):,}장")

    assign = assign_combos(combo_classes, ratios, rng)

    all_splits = {"train": [], "val": [], "test": []}
    for it in items:
        all_splits[assign[it["combo"]]].append(it)

    combo_to_split: dict[str, str] = {}
    leak = 0
    for split_name in ("train", "val", "test"):
        for item in all_splits[split_name]:
            prev = combo_to_split.get(item["combo"])
            if prev is not None and prev != split_name:
                leak += 1
            combo_to_split[item["combo"]] = split_name

    n_tr, n_va, n_te = (len(all_splits[s]) for s in ("train", "val", "test"))
    total = n_tr + n_va + n_te
    classes_per_split = {
        s: len({it["class_idx"] for it in all_splits[s]})
        for s in ("train", "val", "test")
    }

    print("\n" + "=" * 55)
    print("  분할 결과")
    print("=" * 55)
    print(f"  train: {n_tr:>7,}장  ({n_tr/total*100:4.1f}%)  | 클래스 {classes_per_split['train']}")
    print(f"  val  : {n_va:>7,}장  ({n_va/total*100:4.1f}%)  | 클래스 {classes_per_split['val']}")
    print(f"  test : {n_te:>7,}장  ({n_te/total*100:4.1f}%)  | 클래스 {classes_per_split['test']}")
    print(f"  합계 : {total:>7,}장")
    print(f"  combo 단위 누수(split 간 중복 combo): {leak}건  ← 0 이어야 정상")
    print("=" * 55)

    if leak != 0:
        raise RuntimeError(f"데이터누수 감지: {leak}건의 combo가 여러 split에 걸쳐 있습니다.")
    for s in ("train", "val", "test"):
        if classes_per_split[s] != len(class_dirs):
            print(f"  ⚠ 경고: {s} split 에 빠진 클래스가 있습니다 "
                  f"({classes_per_split[s]}/{len(class_dirs)}).")

    if dry_run:
        print("\n[분할] DRY-RUN — splits.json 저장 안 함")
        return

    payload = {
        "meta": {
            "seed": seed,
            "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
            "group_key": "combo (파일명의 '_0_' 앞부분)",
            "counts": {"train": n_tr, "val": n_va, "test": n_te, "total": total},
            "num_classes": len(class_dirs),
            "leak_check": leak,
        },
        "train": all_splits["train"],
        "val":   all_splits["val"],
        "test":  all_splits["test"],
    }
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"\n[분할] 저장 완료: {SPLITS_PATH}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="combo 그룹 단위 train/val/test 분할")
    p.add_argument("--train", type=float, default=0.70, help="train 비율")
    p.add_argument("--val",   type=float, default=0.15, help="val 비율")
    p.add_argument("--test",  type=float, default=0.15, help="test 비율")
    p.add_argument("--seed",  type=int,   default=42,   help="난수 시드")
    p.add_argument("--dry-run", action="store_true", help="저장 없이 통계만 출력")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run((a.train, a.val, a.test), a.seed, a.dry_run)
