# Code Drift losses on official DAC

공식 [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)의
`scripts/train.py`에 **기존 DAC 손실 + 선택한 Code Drift 손실 하나**를 연결했습니다.
Code Drift 저자의 공식 코드가 아니라 논문 §IV에 대한 구현입니다.
공식 DAC 설명은 [README_DAC.md](README_DAC.md), 원본 버전은 [UPSTREAM.md](UPSTREAM.md)에 있습니다.

## 먼저 볼 파일

| 파일 | 역할 |
|---|---|
| [scripts/train.py](scripts/train.py) | 공식 학습 루프, 손실 합산, 학습 대상 선택, 저장·재개 |
| [dac/nn/codedrift.py](dac/nn/codedrift.py) | `enc / proj / code` 수식, RVQ 동결, 왕복 경로 |
| [dac/model/dac.py](dac/model/dac.py) | 선택적인 학습용 forward 분기. 파라미터 구조는 동일 |
| [dac/nn/quantize.py](dac/nn/quantize.py) | 수정하지 않은 공식 RVQ: 검색·residual·STE |
| [conf/code_drift](conf/code_drift) | 대조군과 손실별 fine-tuning 예시 설정 |
| [docs/implementation.md](docs/implementation.md) | 수식↔코드, gradient, 원 논문과 구현 선택의 구별 |
| [docs/validation.md](docs/validation.md) | 실제 검사와 검증 범위 |

## 설치

새 환경에서 설치합니다. 기존 연구 환경의 `dac` 설치를 덮어쓸 필요는 없습니다.
Python 3.10, PyTorch/torchaudio 2.7.1에서 CPU 검사를 수행했습니다.
PyTorch/torchaudio는 사용하는 GPU에 맞는 동일 버전 조합을 먼저 설치합니다.

```bash
git clone https://github.com/kimdonggyu2008/code_drift.git
cd code_drift
python -m venv .venv
source .venv/bin/activate
# Install a matching torch/torchaudio build for your machine first.
pip install -e .
pip install tensorboard pytest
```

## 학습

`conf/code_drift/base.yml`의 `train/build_dataset.folders`와 `val/build_dataset.folders`를
실제 데이터 경로로 바꿉니다. 예시 경로에 데이터는 없습니다. G와 D는 가능하면
**같은 사전학습 실행에서 저장한 가중치**로 시작합니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --args.load conf/code_drift/code.yml --pretrained_generator checkpoints/generator/weights.pth --pretrained_discriminator checkpoints/discriminator/weights.pth --code_drift.scope both --save_path runs/ed-code-lambda1
```

Generator는 공식 DAC `weights.pth` 형식(`state_dict`, `metadata`)을 받습니다.
공식 공개 G만 있다면 `--pretrained_discriminator`를 생략할 수 있지만 D는 새로 초기화되며
시작 로그에 표시됩니다. 별도의 D warm-up을 자동으로 수행하지 않습니다. 이 조건을 저자의
사전학습 G+D에서 연속 학습한 조건과 같다고 간주하지 않습니다.

| 비교 | 설정 파일 | 추가 손실 계수 |
|---|---|---:|
| Base 대조군 | `conf/code_drift/base.yml` | 0 |
| Encoder consistency | `conf/code_drift/enc.yml` | 1 |
| Projected consistency | `conf/code_drift/proj.yml` | 10 |
| Code consistency, 로컬 시작값 | `conf/code_drift/code.yml` | 1 |
| Code consistency, 논문 계수 | `conf/code_drift/code_paper_weight.yml` | 100 |

계수는 `--code_drift.weight 0.1`처럼 변경합니다. `code.yml`의 1은 논문 계수 100과
구분한 로컬 시작값입니다. 특정 checkpoint·데이터에서 품질을 유지하는 값은 검증해야 합니다.

학습 대상은 `--code_drift.scope encoder`, `decoder`, `both`입니다. 세 경우 모두
**RVQ의 in_proj·embedding·out_proj를 고정**하고 dropout을 끕니다. Discriminator는 모두
학습합니다. 각 범위마다 Base와 Code를 짝지어 같은 G/D 출발점, 데이터, seed, 학습량으로 비교합니다.

기본 예시는 44.1 kHz, 9 codebooks, batch 4, 50,000 updates입니다. 원래 DAC의
0.38초 학습 crop, 전처리, 손실 설정을 상속합니다. 원래 DAC 학습은 기존
`conf/final/44khz.yml`로 실행합니다. `code_drift.enabled`의 기본값은 false이므로
RVQ도 원래 방식으로 학습됩니다. `scripts/train_no_adv.py`에는 새 기능을 넣지 않았습니다.

## 기존 음질 손실

예시 설정의 generator 목적함수는 다음과 같습니다.

```text
15 * mel + 1 * adversarial + 2 * feature_matching
 + 0.25 * commitment + 1 * codebook
 + lambda * selected_Code_Drift_loss
```

RVQ가 고정되면 codebook loss는 계산·기록되지만 frozen codebook을 갱신하지 않습니다.
PESQ는 학습 목적함수에 직접 들어가지 않습니다. STFT/waveform loss도 공식 trainer처럼
기록하지만, `lambdas`에 없는 항목은 최적화 합계에 넣지 않습니다.

TensorBoard에 원래 항목과 `codedrift/base_loss`, 선택한 raw loss,
`codedrift/weighted_loss`, 최종 `loss`가 따로 기록됩니다.

```bash
tensorboard --logdir runs
```

## 중단 후 재개

동일한 설정·범위·계수·`save_path`를 사용하고, 최초 가중치 옵션 대신 `--resume`를 붙입니다.
`num_iters`는 **추가 횟수가 아닌 전체 목표 update 수**입니다.

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train --args.load conf/code_drift/code.yml --code_drift.scope both --save_path runs/ed-code-lambda1 --num_iters 50000 --resume
```

검증 시점과 마지막 update에 `latest/`, 필요시 `best/`, `10k/25k/50k/`를 저장합니다.
G/D, 두 optimizer·scheduler, tracker, Code Drift 설정을 복원합니다. 재개 시
범위·손실·계수가 다르면 중단합니다. 최근 저장 이후의 작업은 재실행해야 합니다.
RNG·데이터 worker 상태의 exact replay는 포함하지 않으므로 중단하지 않은 실행과
bitwise 동일하다고 주장하지 않습니다. 신뢰하는 checkpoint만 로드합니다.

## 검사

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_codedrift.py tests/test_codedrift_cli.py -q
```

실제 gradient, 파라미터 갱신, RVQ 동결, 원본 forward, 기본 손실 합산, CLI 저장·재개를
CPU에서 검사합니다. 다운로드한 모델이나 GPU 본학습은 이 검사에 필요하지 않습니다.

이 저장소는 **손실 구현과 학습 연결의 공유본**입니다. 기존 연구의 mixed-data sampler,
D warm-up, 결과 CSV, 평가기 32종, W&B/Fibery 자동화와 학습 결과는 포함하지 않습니다.
이 새 trainer가 과거 실험 전체를 그대로 재생한다고 주장하지 않습니다.

## 출처

- [Code Drift: Towards Idempotent Neural Audio Codecs, arXiv v2](https://arxiv.org/abs/2410.11025v2)
- [High-Fidelity Audio Compression with Improved RVQGAN](https://arxiv.org/abs/2306.06546)
- [공식 DAC 코드](https://github.com/descriptinc/descript-audio-codec)

공식 DAC의 MIT [LICENSE](LICENSE)를 유지합니다. 추가한 코드에도 같은 라이선스를 적용합니다.
