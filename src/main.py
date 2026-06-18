"""
main.py
=======
DrugCNN 병용금기 탐지 파이프라인 엔트리포인트.

사용 방법 (순서대로 실행)

  [Step 1] 전처리
      python src/main.py preprocess

  [Step 2] 사진(combo) 단위 분할 — train/val/test (누수 차단)
      python src/main.py split

  [Step 3-A] 소규모 하이퍼파라미터 탐색 (10~20 클래스, 15~20 epoch)
      python src/main.py train --classes 20 --epochs 20 --lr 1e-3
      python src/main.py train --classes 20 --epochs 20 --lr 3e-4

  [Step 3-B] 전체 학습 (100 클래스, 불균형 보정 — 기본 weighted loss)
      python src/main.py train

  [Step 4] test 전용 최종 평가 (혼동행렬·Macro F1·Top-5)
      python src/main.py evaluate

  [Step 5] 추론 + 병용금기 체크
      python src/main.py predict --images drug1.png drug2.png
      python src/main.py predict --photo combo.png --json combo.json
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent


def run(script: str, extra_args: list) -> None:
    cmd = [sys.executable, str(SRC / script)] + extra_args
    print(f"\n▶ 실행: {' '.join(cmd)}\n{'─'*55}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[오류] {script} 실패 (exit code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DrugCNN Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preprocess", help="Step 1: 이미지 전처리 (bbox 크롭)")

    s = sub.add_parser("split", help="Step 2: 사진(combo) 단위 train/val/test 분할")
    s.add_argument("--train", type=float, default=0.70)
    s.add_argument("--val",   type=float, default=0.15)
    s.add_argument("--test",  type=float, default=0.15)
    s.add_argument("--seed",  type=int,   default=42)

    t = sub.add_parser("train", help="Step 3: 모델 학습 (불균형 보정)")
    t.add_argument("--lr",       type=float, default=1e-3)
    t.add_argument("--batch",    type=int,   default=128)
    t.add_argument("--dropout",  type=float, default=0.5)
    t.add_argument("--epochs",   type=int,   default=80)
    t.add_argument("--patience", type=int,   default=8)
    t.add_argument("--classes",  type=int,   default=100)
    t.add_argument("--loss",     choices=["ce", "weighted", "focal"], default="weighted")
    t.add_argument("--sampler",  choices=["none", "weighted"], default="none")
    t.add_argument("--gamma",    type=float, default=2.0)

    e = sub.add_parser("evaluate", help="Step 4: test 전용 최종 평가")
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--batch", type=int, default=256)

    p = sub.add_parser("predict", help="Step 5: 추론 + 병용금기 체크")
    p.add_argument("--images",     nargs="+")
    p.add_argument("--photo")
    p.add_argument("--json")
    p.add_argument("--checkpoint", default=None)

    args, _ = parser.parse_known_args()

    if args.command == "preprocess":
        run("1_preprocess.py", [])

    elif args.command == "split":
        run("0_split.py", [
            "--train", str(args.train), "--val", str(args.val),
            "--test", str(args.test), "--seed", str(args.seed),
        ])

    elif args.command == "train":
        extra = [
            "--lr",       str(args.lr),
            "--batch",    str(args.batch),
            "--dropout",  str(args.dropout),
            "--epochs",   str(args.epochs),
            "--patience", str(args.patience),
            "--classes",  str(args.classes),
            "--loss",     args.loss,
            "--sampler",  args.sampler,
            "--gamma",    str(args.gamma),
        ]
        run("3_train.py", extra)

    elif args.command == "evaluate":
        extra = ["--batch", str(args.batch)]
        if args.checkpoint:
            extra += ["--checkpoint", args.checkpoint]
        run("5_evaluate.py", extra)

    elif args.command == "predict":
        extra = []
        if args.images:
            extra += ["--images"] + args.images
        elif args.photo and args.json:
            extra += ["--photo", args.photo, "--json", args.json]
        if args.checkpoint:
            extra += ["--checkpoint", args.checkpoint]
        run("4_predict_and_check.py", extra)


if __name__ == "__main__":
    main()
