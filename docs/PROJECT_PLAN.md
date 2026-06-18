# 💊 병용금기 약물 안내 모델 — 프로젝트 계획서

> **목표:** 사진 속 약물을 CNN으로 분류하고, 병용금기 DB와 대조해 위험한 약물 조합을 탐지한다.

---

## ✅ 지금까지 완료한 작업

### 1. 데이터 준비

- **이미지 데이터** `data/images/TS_1 ~ TS_8`
  - 총 46,117개 JSON 라벨 파싱 완료
  - 고유 약물 코드 **118개** 확인
  - 각 이미지에 4개 약물이 함께 촬영된 구조 (COCO bbox 포맷)

- **병용금기 DB** `data/contraindicated_drugs.xlsx`
  - 총 **1,628쌍** 로드 완료
  - 컬럼: `num`, `material_1`, `material_2`, `note`, `information`

### 2. 분석 스크립트 작성

| 파일 | 역할 | 상태 |
|------|------|------|
| `src/examination/find_contraindicated.py` | 데이터셋 × 병용금기 DB 교차 검색 | ✅ 완료 |
| `src/examination/compare.py` | 라벨 JSON 성분명 vs CSV 정합성 검증 | ✅ 완료 (경로 수정 필요) |
| `src/plot/drug_drug_interactions.py` | 병용금기 관계 MLP 스타일 시각화 | ✅ 완료 |

### 3. 교차 검색 결과 (`output/contraindicated_in_dataset.csv`)

데이터셋 118개 약물 중 **2쌍**의 병용금기 발견:

| 성분 1 | 성분 2 | 약물명 1 | 약물명 2 | 위험 |
|--------|--------|----------|----------|------|
| rasagiline | tramadol | 아질렉트정 | 울트라셋이알서방정 | 세로토닌증후군 |
| pseudoephedrine | rasagiline | 액티프롤정 60mg | 아질렉트정 | 고혈압/빈맥 |

> **핵심 인사이트:** 아질렉트정(rasagiline)이 두 쌍 모두에 관여 — 학습 클래스에 반드시 포함 권장

---

## 🔲 앞으로 할 작업

### Phase 1 — 데이터 전처리

- [ ] **`src/1_preprocess.py` 작성**
  - COCO bbox 좌표를 이용해 4약물 사진에서 개별 약물 이미지 크롭
  - 출력: 단일 약물 이미지 약 60,000장 (`128×128` 리사이즈)
  - 크롭 방식: 직사각형 bbox (정밀 누끼 불필요)

- [ ] **100개 학습 클래스 선정**
  - 병용금기 쌍에 등장하는 약물 우선 포함 (아질렉트정 등 3개 확정)
  - 나머지는 데이터 수량 기준으로 상위 클래스 선택
  - `compare.py` 경로 수정: `contraindicated_drugs(old).csv` → `contraindicated_drugs.xlsx`

### Phase 2 — 모델 구현

- [ ] **`src/2_model.py` 작성 — DrugCNN 정의**
  - 구조: Conv/BN/ReLU/MaxPool × 4블록 + FC + Dropout + Softmax
  - 파라미터 수 목표: ~8.8M
  - 입력: `(batch, 3, 128, 128)` / 출력: `(batch, 100)`

- [ ] **`src/3_train.py` 작성 — 학습 파이프라인**
  - 손실함수: CrossEntropyLoss
  - 옵티마이저: Adam
  - Early stopping (patience=8)
  - 학습 곡선 저장 (`output/loss_curve.png`)

### Phase 3 — 하이퍼파라미터 튜닝 (2단계 전략)

> 전체 100 클래스 훈련 전, 소규모 실험으로 후보를 빠르게 제거

- [ ] **1단계 — 소규모 탐색 (10~20 클래스, 15~20 epochs)**
  - 탐색 항목: learning rate, batch size, dropout rate, augmentation 조합
  - GPU: RTX 4090 (비용 대비 속도 최적)

- [ ] **2단계 — 최적 조합으로 전체 훈련 (100 클래스, ~80 epochs)**
  - Early stopping으로 실제 수렴 epoch 자동 결정

### Phase 4 — 추론 + 병용금기 체크

- [ ] **`src/4_predict_and_check.py` 작성**
  - CNN 추론 → 약물 코드 → 성분명(영문) 변환
  - 변환된 성분명을 `find_contraindicated.py` 로직으로 DB 조회
  - 입력된 약물 세트 내 병용금기 쌍 전부 출력
  - 출력 예시:
    ```
    [경고] 아질렉트정 + 울트라셋이알서방정
           → 세로토닌증후군 발생 위험 증가
    ```

### Phase 5 — 마무리 및 발표 준비

- [ ] 최종 파이프라인 통합 (`src/main.py` 엔트리포인트 작성)
- [ ] 결과 시각화 업데이트 (`drug_drug_interactions.py` 최종 실행)
- [ ] README.md 업데이트 (실제 디렉터리 구조 반영)
- [ ] 발표 자료 작성

---

## 📁 현재 디렉터리 구조

```
team_project_06/
├── data/
│   ├── images/         TS_1 ~ TS_8  (원본 이미지)
│   ├── labels/         TL_1 ~ TL_8  (COCO JSON 라벨)
│   └── contraindicated_drugs.xlsx   (병용금기 1,628쌍)
│
├── src/
│   ├── main.py                      (엔트리포인트 — 미작성)
│   ├── examination/
│   │   ├── compare.py               (정합성 검증)
│   │   └── find_contraindicated.py  (교차 검색 ✅)
│   ├── plot/
│   │   └── drug_drug_interactions.py (시각화)
│   ├── module/                      (재사용 모듈 — 미작성)
│   └── test/
│       └── test.ipynb
│
└── output/
    ├── contraindicated_in_dataset.csv  (교차 검색 결과 ✅)
    └── Figure_1.png
```

---

## 🔧 알려진 버그 / 수정 필요 항목

| 파일 | 문제 | 해결 방법 |
|------|------|----------|
| `src/examination/compare.py` | `contraindicated_drugs(old).csv` 경로 참조 (파일 없음) | 경로를 `contraindicated_drugs.xlsx`로 수정, `read_excel`로 변경 |
| `src/plot/drug_drug_interactions.py` | `material_1` 컬럼을 `list.split('(')`로 처리해 집계 오류 가능 | `clean_excel_ingredient()` 함수로 교체 |
