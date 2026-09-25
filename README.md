# Transformer Practice

논문 **"Attention Is All You Need"** (Vaswani et al., 2017)를 코드로 직접 따라가며 공부하는 연습 레포입니다.
딥러닝 코드를 처음 짜보는 상태에서 시작했기 때문에, PyTorch 기초 → 참고 구현 읽기 → 사전학습 모델로 추론 → **Transformer를 직접 구현해서 번역 학습**의 순서로 진행합니다.

## 학습 계획

1. **파이썬 클래스 / PyTorch 기초** 익히기 (`class_basics.py`)
2. **참고 구현 읽기**: Harvard NLP의 *The Annotated Transformer* 핵심 부분을 CPU에서 돌아가게 정리 (`transformer_from_scratch.py`)
3. **사전학습 모델로 추론 + attention 시각화** (`pretrained_inference.py`, `visualize_attention.py`)
4. **직접 구현** (`toy_translation.ipynb`): 아래에서 위로(bottom-up) 부품을 하나씩 짜서 조립
   - **로컬(노트북, CPU)**: 짧은 문장 몇 개로 shape / 학습 / 추론이 돌아가는지만 확인 (같은 문장을 외워서 번역해내면 통과)
   - **Colab(GPU)**: 같은 노트북으로 공개 데이터셋 **Multi30k (EN→DE)** 를 학습하고 BLEU로 평가

## 진행 상황

- [x] 파이썬 클래스 / `nn.Module` / 학습 루프 기초 (`class_basics.py`, 예제 7 정확도 100%)
- [x] 참고 구현 실행: 복사(copy) 과제 학습 → 입력 그대로 복원
- [x] 사전학습 번역 모델 추론, encoder self-attention 히트맵
- [x] 직접 구현: 데이터 파이프라인 (토큰화, 어휘, 패딩 `collate`)
- [x] 직접 구현: 마스크 3종 (`make_pad_mask`, `subsequent_mask`, `make_tgt_mask`)
- [x] 직접 구현: `attention`, `MultiHeadAttention`, `Transformer` 뼈대
- [ ] `PositionwiseFFN`, `PositionalEncoding`, `EncoderLayer`, `DecoderLayer`
- [ ] 학습 루프(`train`), 추론(`greedy_decode`), 평가(`evaluate`)
- [ ] 로컬 과적합 테스트 → Colab에서 Multi30k 학습 / BLEU

## 폴더 구성

| 파일 | 설명 |
|---|---|
| `toy_translation.ipynb` | **직접 구현 중인 메인 노트북.** Part A 데이터 → Part B 모델 → Part C 학습 → Part D 추론/평가. 미구현 셀은 `NotImplementedError`로 남아 있음 |
| `transformer_from_scratch.py` | 논문 구조를 그대로 옮긴 순수 PyTorch 참고 구현 (Annotated Transformer에서 무거운 의존성 제거). 실행하면 복사 과제를 학습 |
| `study.ipynb` | 위 참고 구현을 셀 단위로 만져보는 노트북 |
| `pretrained_inference.py` | `Helsinki-NLP/opus-mt-en-de`(encoder-decoder, 6층 8헤드)로 영→독 번역 추론 |
| `visualize_attention.py` | 위 모델의 encoder self-attention 히트맵 생성 → `attention_heatmap.png` |
| `class_basics.py` | 파이썬 클래스 문법 + `nn.Module` 연습 예제 (예제 7: 패딩이 섞인 시퀀스 분류기) |
| `requirements.txt` | 로컬 환경 패키지 목록 |

## 환경 설정

Windows + Python 3.13에서 CPU만으로 진행했습니다.

```bash
python -m venv venv
source venv/Scripts/activate            # PowerShell: venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece sacremoses numpy matplotlib jupyter
```

`requirements.txt`의 `torch==...+cpu`는 PyPI가 아니라 PyTorch CPU 인덱스에서 받아야 하므로 위처럼 torch만 따로 설치하세요.
노트북을 VS Code에서 쓰려면 venv를 Jupyter 커널로 등록합니다.

```bash
python -m ipykernel install --user --name transformer-study --display-name "Transformer Study (venv)"
```

## 실행 방법

```bash
python class_basics.py                  # 파이썬 클래스 / nn.Module 예제
python transformer_from_scratch.py      # 복사 과제 학습 (CPU 약 2분)
python pretrained_inference.py          # 사전학습 번역 모델 추론 (최초 1회 가중치 다운로드)
python visualize_attention.py           # attention_heatmap.png 생성
```

### 직접 구현 노트북 (`toy_translation.ipynb`)

- `MODE = "local"`(기본): `TINY_TEXT`의 짧은 문장 8개로 학습/추론 확인. 학습·평가 셀이 완성되면 loss가 0에 가깝게 내려가고 같은 문장을 그대로 번역해야 합니다.
- `MODE = "colab"`: Colab에서 노트북을 열고 GPU 런타임을 선택한 뒤 실행합니다.
  ```python
  !pip install datasets sacrebleu
  ```
  `load_multi30k()`가 Hugging Face의 `bentrevett/multi30k`를 불러옵니다.
  - 데이터 파이프라인은 실제 Multi30k 학습 파일로 미리 확인했습니다: 29,000문장 중 30토큰 이하 28,908개, 어휘 `min_freq=2, max 8000` 기준 영어 5,871 / 독일어 7,833, 타겟 UNK 비율 약 2.7%.
  - `load_dataset` 호출과 GPU 학습 자체는 **아직 Colab에서 검증하지 않았습니다.**
  - GPU에서는 `subsequent_mask`를 `tgt.device`로 옮겨야 `make_tgt_mask`의 `&`가 동작합니다 (노트북에 메모 있음).

## 구현하며 정리한 핵심 개념

- **마스크는 세 곳에서 쓰인다**: 인코더 self-attention(`src` pad), 디코더 masked self-attention(`tgt` pad + 미래), 디코더 cross-attention(`memory`의 `src` pad).
- 마스크는 임베딩 전 **id 행렬**로 만들고, attention 점수표에서 `masked_fill(mask == 0, -1e9)`로 소비된다. (평균 풀링 예제처럼 벡터를 0으로 만드는 것과 다름)
- `attention`은 헤드 차원 `(B, h)`를 배치로 취급해 모든 헤드를 한 번에 계산하고, 헤드를 쪼개고 합치는 일은 `MultiHeadAttention`이 담당한다.
- attention의 `√d_k`로 나누기(3.2.1절)와 임베딩의 `√d_model` 곱하기(3.4절)는 서로 다른 장치다.
- `CrossEntropyLoss`에는 softmax 전의 raw logits와 `long` 타입 정답 `(N,)`을 넣는다.

## 참고 자료

- 논문: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://github.com/harvardnlp/annotated-transformer) (Harvard NLP). 이 레포에는 포함하지 않았고, 필요하면 별도로 clone하세요.
- 데이터셋: [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)
