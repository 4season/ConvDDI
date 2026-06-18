# 🔬 DrugCNN 기술 가이드 — 크롭 · 모델 · 하이퍼파라미터

> 이 문서는 병용금기 약물 분류 파이프라인의 핵심 3단계를 수식과 함께 상세히 설명합니다.

---

## 목차

1. [크롭 — 개별 약물 이미지 추출](#1-크롭--개별-약물-이미지-추출)
2. [모델 구현 — DrugCNN 구조](#2-모델-구현--drugcnn-구조)
3. [하이퍼파라미터 — 학습 전략](#3-하이퍼파라미터--학습-전략)

---

## 1. 크롭 — 개별 약물 이미지 추출

### 1-1. 왜 크롭인가?

원본 사진 1장에는 **약물 4개가 함께 촬영**되어 있습니다.  
CNN이 "이 이미지는 약물 X다"라고 학습하려면, 각 약물을 **개별 이미지로 분리**해야 합니다.

```
원본 이미지 (976 × 1280)
┌──────────────────────────────┐
│  [약A]   [약B]               │
│                              │  →  크롭  →  약A 단독 이미지
│  [약C]   [약D]               │              약B 단독 이미지 ...
└──────────────────────────────┘
```

### 1-2. COCO Bounding Box 포맷

데이터셋 JSON의 `annotations` 필드:

```json
"bbox": [x_min, y_min, width, height]
```

| 값 | 의미 |
|----|------|
| $x_{min}$ | 박스 좌상단의 x 픽셀 좌표 |
| $y_{min}$ | 박스 좌상단의 y 픽셀 좌표 |
| $w$ | 박스 너비 (픽셀) |
| $h$ | 박스 높이 (픽셀) |

### 1-3. 크롭 수식

원본 이미지를 $I \in \mathbb{R}^{H \times W \times 3}$ 이라 하면, 크롭된 이미지는:

$$I_{crop} = I\bigl[y_{min} : y_{min} + h,\quad x_{min} : x_{min} + w\bigr]$$

NumPy/PIL 인덱싱에 대응하면:

$$I_{crop}[r, c] = I[y_{min} + r,\quad x_{min} + c], \quad 0 \le r < h,\quad 0 \le c < w$$

### 1-4. 리사이즈 — 128 × 128

크롭된 이미지는 크기가 제각각이므로, CNN 입력 통일을 위해 **128 × 128**으로 리사이즈합니다.  
PyTorch `transforms.Resize`는 내부적으로 **이중선형 보간(Bilinear Interpolation)** 을 사용합니다.

목표 좌표 $(x', y')$에서의 픽셀값을 원본 좌표로 역매핑:

$$x = x' \cdot \frac{w}{128}, \qquad y = y' \cdot \frac{h}{128}$$

정수 좌표가 아닌 경우, 주변 4개 픽셀의 가중 평균으로 보간:

$$I_{out}(x', y') = \sum_{i \in \{0,1\}} \sum_{j \in \{0,1\}} w_{ij} \cdot I(\lfloor x \rfloor + i,\; \lfloor y \rfloor + j)$$

여기서 가중치:

$$w_{00} = (1 - \Delta x)(1 - \Delta y), \quad \Delta x = x - \lfloor x \rfloor, \quad \Delta y = y - \lfloor y \rfloor$$

### 1-5. 정규화 (Normalization)

픽셀값을 $[0, 255]$에서 $[0.0, 1.0]$으로 스케일 후, ImageNet 통계로 표준화:

$$\hat{x}_c = \frac{x_c / 255 - \mu_c}{\sigma_c}$$

| 채널 | $\mu_c$ | $\sigma_c$ |
|------|---------|------------|
| R | 0.485 | 0.229 |
| G | 0.456 | 0.224 |
| B | 0.406 | 0.225 |

> **왜 ImageNet 통계?** 우리 데이터셋이 약물 이미지에 특화되어 있더라도, 약물 자체는 실세계 색상을 가지므로 ImageNet 통계를 사용하는 것이 수렴에 유리합니다.

---

## 2. 모델 구현 — DrugCNN 구조

### 2-1. 전체 아키텍처 개요

```
입력 (B, 3, 128, 128)
   │
   ▼
Conv Block 1 → (B, 32, 64, 64)
   ▼
Conv Block 2 → (B, 64, 32, 32)
   ▼
Conv Block 3 → (B, 128, 16, 16)
   ▼
Conv Block 4 → (B, 256, 8, 8)
   ▼
Flatten → (B, 16384)
   ▼
FC1 + Dropout → (B, 512)
   ▼
FC2 + Softmax → (B, 100)
```

각 Conv Block = **Conv2d → BatchNorm → ReLU → MaxPool**

---

### 2-2. Convolution 레이어

#### 연산 정의

입력 특징맵 $X \in \mathbb{R}^{C_{in} \times H \times W}$, 필터 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$ 에 대해:

$$Y[c_{out}, i, j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} W[c_{out}, c_{in}, m, n] \cdot X[c_{in},\; i+m,\; j+n] + b[c_{out}]$$

#### 출력 크기 공식

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

| 기호 | 의미 |
|------|------|
| $P$ | 패딩 (padding) |
| $K$ | 커널 크기 (kernel size) |
| $S$ | 스트라이드 (stride) |

본 프로젝트에서는 $K=3, P=1, S=1$ 로 설정 → $H_{out} = H_{in}$ (크기 유지)

#### 파라미터 수

$$\text{params}_{conv} = K^2 \times C_{in} \times C_{out} + C_{out}$$

---

### 2-3. Batch Normalization

미니배치 $\mathcal{B} = \{x_1, \ldots, x_m\}$에 대해:

**① 평균:**

$$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

**② 분산:**

$$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**③ 정규화:**

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

**④ 스케일 & 이동 (학습 파라미터 $\gamma, \beta$):**

$$y_i = \gamma \hat{x}_i + \beta$$

> **역할:** 각 레이어의 입력 분포를 안정화시켜 그레이디언트 소실 문제를 완화하고 학습 속도를 높입니다.

---

### 2-4. ReLU 활성화 함수

$$f(x) = \max(0,\; x)$$

미분:

$$f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}$$

> **왜 ReLU?** Sigmoid/Tanh 대비 그레이디언트 소실이 없고, 연산이 단순해 학습이 빠릅니다.

---

### 2-5. Max Pooling

$K_{pool} = 2, S_{pool} = 2$ (2×2 풀링)에서 출력:

$$Y[i, j] = \max_{0 \le m, n < K_{pool}} X[i \cdot S_{pool} + m,\quad j \cdot S_{pool} + n]$$

출력 크기:

$$H_{out} = \left\lfloor \frac{H_{in}}{K_{pool}} \right\rfloor$$

따라서 128 → 64 → 32 → 16 → 8 (4회 반복)

---

### 2-6. Dropout

학습 시, 각 뉴런을 확률 $p$로 무작위 비활성화:

$$\tilde{x}_i = \begin{cases} 0 & \text{확률 } p \text{ 로} \\ \dfrac{x_i}{1-p} & \text{확률 } 1-p \text{ 로} \end{cases}$$

$\frac{1}{1-p}$ 로 스케일하는 이유 — 학습/추론 시 **기댓값 보존**:

$$\mathbb{E}[\tilde{x}_i] = (1-p) \cdot \frac{x_i}{1-p} + p \cdot 0 = x_i$$

> **추론 시** Dropout은 비활성화되고 원본 $x_i$를 그대로 사용합니다.

---

### 2-7. Softmax 출력층

$C = 100$개 클래스에 대한 로짓 벡터 $z \in \mathbb{R}^{100}$:

$$\text{softmax}(z)_k = \frac{e^{z_k}}{\displaystyle\sum_{j=1}^{C} e^{z_j}}, \qquad k = 1, \ldots, C$$

출력은 각 클래스의 **확률** ($\sum_k \text{softmax}(z)_k = 1$).

**수치 안정성** 처리 (PyTorch 내부):

$$\text{softmax}(z)_k = \frac{e^{z_k - \max(z)}}{\displaystyle\sum_{j=1}^{C} e^{z_j - \max(z)}}$$

---

### 2-8. 파라미터 수 계산

| 레이어 | 수식 | 파라미터 수 |
|--------|------|------------|
| Conv Block 1 | $3^2 \times 3 \times 32 + 32$ | 896 |
| BN 1 | $2 \times 32$ | 64 |
| Conv Block 2 | $3^2 \times 32 \times 64 + 64$ | 18,496 |
| BN 2 | $2 \times 64$ | 128 |
| Conv Block 3 | $3^2 \times 64 \times 128 + 128$ | 73,856 |
| BN 3 | $2 \times 128$ | 256 |
| Conv Block 4 | $3^2 \times 128 \times 256 + 256$ | 295,168 |
| BN 4 | $2 \times 256$ | 512 |
| FC1 | $16384 \times 512 + 512$ | 8,389,120 |
| FC2 | $512 \times 100 + 100$ | 51,300 |
| **합계** | | **≈ 8.83M** |

> Flatten 직전 특징맵: $(B, 256, 8, 8)$ → $256 \times 8 \times 8 = 16{,}384$

---

## 3. 하이퍼파라미터 — 학습 전략

### 3-1. 손실 함수 — Cross Entropy Loss

배치 크기 $N$, 정답 클래스 $y_i$, 모델 예측 확률 $\hat{p}_{i,k}$에 대해:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{p}_{i,\, y_i}$$

Softmax 확률을 대입하면:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left( z_{i,y_i} - \log \sum_{j=1}^{C} e^{z_{i,j}} \right)$$

> **직관:** 정답 클래스의 확률이 낮을수록 $-\log(\hat{p})$는 커집니다 ($-\log(1)=0,\; -\log(0.1) \approx 2.3$).

---

### 3-2. 옵티마이저 — Adam

Adam은 각 파라미터마다 **적응적 학습률**을 유지합니다.

**① 1차 모멘트 (그레이디언트의 지수이동평균):**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t$$

**② 2차 모멘트 (그레이디언트 제곱의 지수이동평균):**

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2$$

**③ 편향 보정 (초기 $t$가 작을 때 $m_t, v_t$가 0에 편향되는 문제 해결):**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**④ 파라미터 업데이트:**

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\, \hat{m}_t$$

| 하이퍼파라미터 | 기본값 | 역할 |
|---------------|--------|------|
| $\eta$ (learning rate) | 1e-3 | 전체 학습 속도 |
| $\beta_1$ | 0.9 | 모멘텀 감쇠율 |
| $\beta_2$ | 0.999 | RMSProp 감쇠율 |
| $\epsilon$ | 1e-8 | 0 나눗셈 방지 |

---

### 3-3. Learning Rate

학습률 $\eta$는 손실 지형에서 한 스텝의 크기입니다.

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$

| 값 | 결과 |
|----|------|
| 너무 크면 ($\eta \gg 1$) | 손실이 발산 (overshooting) |
| 너무 작으면 ($\eta \ll 0.0001$) | 수렴이 매우 느림 |
| 일반 권장 (Adam) | **1e-3 ~ 1e-4** |

**탐색 전략:** 1단계 소규모 실험(10~20 클래스, 15~20 epoch)에서 아래 후보 비교:

| 후보 | 예상 동작 |
|------|----------|
| 1e-2 | 불안정, 발산 가능성 |
| **1e-3** | 대부분 딥러닝 태스크의 시작점 |
| 3e-4 | 안정적, 다소 느림 |
| 1e-4 | 매우 안정, 수렴 느림 |

---

### 3-4. Batch Size

미니배치 $\mathcal{B}$ 크기 $B$:

$$g_{\mathcal{B}} = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla_\theta \mathcal{L}(x_i, y_i)$$

| 배치 크기 | 특성 |
|----------|------|
| 작음 (16~32) | 노이즈 많음 → 정규화 효과, GPU 메모리 절약 |
| 중간 (64~128) | 균형점, **권장** |
| 큼 (256~512) | 안정적이나 일반화 성능 저하 가능 |

RTX 4090 기준 128×128 이미지에서 **batch_size = 64~128** 이 메모리·속도 균형에 최적입니다.

---

### 3-5. Dropout Rate

FC1 레이어 뒤에 적용. 탈락 확률 $p$:

- $p$ 너무 작 → 과적합(overfitting)
- $p$ 너무 큼 → 과소적합(underfitting), 학습 불안정

| 후보 | 권장 상황 |
|------|----------|
| 0.3 | 데이터 충분, 빠른 학습 |
| **0.5** | 일반적 권장 (Srivastava et al., 2014) |
| 0.7 | 과적합이 심각할 때 |

---

### 3-6. Early Stopping

검증 손실 $\mathcal{L}_{val}$ 이 `patience` epoch 동안 개선되지 않으면 학습 중단:

```
best_val_loss = ∞
patience_counter = 0

매 epoch 끝:
  if L_val < best_val_loss:
    best_val_loss = L_val
    patience_counter = 0
    모델 가중치 저장
  else:
    patience_counter += 1
    if patience_counter >= patience:
      학습 중단 (overfitting 구간 진입)
```

본 프로젝트: `patience = 8`

---

### 3-7. 데이터 증강 (Augmentation)

학습 시 랜덤 변환으로 데이터 다양성 확보:

| 변환 | 수식 / 설명 | 약물 분류 적합성 |
|------|------------|----------------|
| RandomHorizontalFlip | $x \leftarrow W - 1 - x$ (확률 0.5) | ✅ 약물은 좌우 대칭 |
| RandomRotation ($\pm15°$) | 회전 행렬 적용 | ✅ 카메라 각도 다양 |
| ColorJitter | 밝기·대비 ±20% | ✅ 조명 조건 다양 |
| RandomVerticalFlip | | ⚠️ 글자 있는 경우 주의 |
| CenterCrop | | ✅ 박스 여백 제거 |

---

### 3-8. 2단계 하이퍼파라미터 탐색 전략

```
[1단계] 소규모 탐색
  ├─ 클래스 수: 10~20개
  ├─ Epoch: 15~20
  └─ 탐색 격자:
       lr ∈ {1e-3, 3e-4, 1e-4}
       batch ∈ {64, 128}
       dropout ∈ {0.3, 0.5}
         → 조합 총 12가지, 각 약 15분 소요

[2단계] 전체 훈련
  ├─ 클래스 수: 100개
  ├─ Epoch: ~80 (early stopping)
  └─ 1단계 최적 조합 1가지만 사용
         → 약 1~2시간 (RTX 4090 기준)
```

**비교 지표:**

- 검증 정확도 (val accuracy)
- 검증 손실 수렴 속도 (val loss curve 기울기)
- 과적합 시점 (train loss ↓ 인데 val loss ↑ 인 epoch)

---

## 참고 문헌

- Ioffe & Szegedy (2015). *Batch Normalization: Accelerating Deep Network Training.* ICML.
- Srivastava et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR.
- Kingma & Ba (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
