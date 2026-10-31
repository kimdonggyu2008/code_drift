# 수식과 공식 코드의 연결

```text
x → encoder → z0 → original RVQ → zq0 → decoder → x1
                                                    ↓ (detach 없음)
                                                 encoder → z1 → original RVQ
```

`DAC.forward`는 기본 호출에서는 원래 동작을 수행합니다. `code_drift_loss`를 지정한
학습 forward만 `forward_with_code_drift`로 들어갑니다. Helper는 공식 encoder,
quantizer, decoder를 그대로 호출합니다. 두 encoding은 한 forward 안에서 계산합니다.
GPU/DDP 본학습 검증은 별도입니다.

RVQ는 앞 단계의 복원 벡터를 residual에서 빼고 residual을 `in_proj`로 투영합니다.
공식 `latents`는 단계별 투영값을 channel 방향으로 연결한 것입니다. 선택 코드 `e0`는
첫 encoding의 정수 ID로 `decode_code`하여 얻습니다. `out_proj` 이후 벡터를 쓰지 않습니다.

| 손실 | 비교 | 첫 branch gradient |
|---|---|---|
| `enc` | `z1 − z0` | 허용 |
| `proj` | `p1 − p0` | 허용 |
| `code` | `p1 − stop_gradient(e0)` | hard index·목표 코드 벡터로는 보내지 않음 |

각 손실은 `mean_B,T(norm_C(left-right, p=2))`입니다. MSE나 squared L2가 아닙니다.
9×8D 예시는 projected/code space에서 72D channel norm입니다. RVQ 최근접 검색은
공식 코드의 정규화 거리를 사용하지만, 추가 손실은 정규화 전 projection/embedding을 비교합니다.

## 파라미터와 gradient

| scope | Encoder | RVQ in/out projection·embedding | Decoder | GAN discriminator |
|---|---|---|---|---|
| `both` | 학습 | 고정 | 학습 | 학습 |
| `encoder` | 학습 | 고정 | 고정 | 학습 |
| `decoder` | 고정 | 고정 | 학습 | 학습 |

`code_drift.enabled=true`일 때 적용됩니다. RVQ 입력 미분이 필요하므로 frozen RVQ를
`no_grad()`로 감싸지 않습니다. Decoder-only에서도 고정 encoder의 입력 미분이 decoder에
전달됩니다. Encoder-only에서도 고정 decoder를 통과하는 미분이 encoder로 전달됩니다.
기본 STE는 그대로입니다. **손실 측정 위치와 파라미터 학습 대상은 다릅니다.**

`model.train()` 이후 고정 RVQ를 다시 eval로 둡니다. Dropout과 불필요한 dropout RNG
소비를 막습니다. `weight=0` 또는 `loss=none`은 두 번째 encoding을 실행하지 않습니다.
학습 중 바뀌는 latent를 데이터셋 전체에서 고정 재사용하지 않습니다.

## train.py 합산

1. 원래 DAC의 reconstruction, commitment/codebook loss를 계산합니다.
2. 원래 discriminator update를 수행합니다.
3. 원래 mel·GAN·feature matching 등을 `lambdas`대로 합산합니다.
4. `base_loss + code_drift.weight * selected_loss`로 generator를 업데이트합니다.

기본 손실을 제거하거나 세 Code Drift 손실을 자동으로 모두 더하지 않습니다.
입력 waveform의 전처리는 원래 trainer가 수행합니다. 반환 waveform에는 새 normalization이나
augmentation을 적용하지 않고, 원래 길이로 자른 뒤 재인코딩합니다.

## 원 논문과 구현 선택

[Code Drift v2 §IV](https://arxiv.org/html/2410.11025v2#S4)는 세 표현 거리, 시간 평균,
quantizer 고정, dropout 해제, 추가 50k 학습을 설명합니다. 저자의 기반 모델은 48 kHz이며,
이 예시는 공개 44.1 kHz 모델에서 사용할 수 있는 코드입니다.

다음은 논문에 코드 수준으로 명시되지 않은 **이 구현의 선택**입니다.

- RVQ stage를 channel로 연결한 뒤 L2를 계산합니다. Stage별 norm 평균과 다릅니다.
- Encoder/projection 손실은 두 branch 모두 미분하고 code target만 detach합니다.
- Padding한 모든 latent frame을 시간 평균에 포함합니다.

기존 로컬 학습의 손실 배선을 이식했으며 저자의 비공개 세부 구현과 동일하다고 증명한 것은
아닙니다. 논문 계수 1/10/100과 로컬 `code.yml`의 λ=1을 구분했습니다.

같은 손실 수치/gradient 경로와 동일 학습 실행의 재현은 구분합니다. 데이터 순서,
G/D 출발 상태, augmentation, optimizer, crop, batch, RNG, 평가 자료도 맞춰야 합니다.
