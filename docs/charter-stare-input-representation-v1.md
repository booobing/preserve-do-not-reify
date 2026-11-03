# Charter — StarE 백본 위의 입력 표현 비교 v1: hyper vs reified (설계 초안)

작성 2026-10-08. **상태: 초안. §0 미확정. Clarification Debt 정지(132.5) 유지.** 구현과 실행은 사용자가 §0을 정하고
정지를 해제한 뒤에만 한다. **어떤 결과도 아직 관측되지 않았다.** 이 문서의 수치는 전부 데이터와 코드를 읽어서 센 것이다.

출발점은 2026-10-08 논문 리뷰의 보강 실험 ①이다. "SOTA를 이기려 하기보다, StarE처럼 공개 코드가 있는 모델에 hyper 입력과
reified 입력을 각각 넣어, **어떤 백본에서도 보존한 쪽이 앞선다**를 보인다." 사용자 지시는 "우선 StarE 하나에 두 입력을 넣어
작은 규모로 돌려 본다"이다.

선행 결과는 5장(`docs/thesis/05-ontology-encoder.md`)의 V2-D다. 우리 인코더에서 WD50K_100 MRR이 `hyper` 0.147,
`reified_binary` 0.118(5/5 seed)이었다. 이 charter는 그 비교를 **우리 인코더가 아닌 외부 백본(StarE)** 위에서 반복한다.
백본이 바뀌어도 순서가 유지되면 "표현의 효과"라는 주장이 강해지고, 뒤집히면 5장의 주장은 "우리 인코더에서의 관찰"로 한정돼야 한다.
어느 쪽이든 결과다.

Ambiguity **MEDIUM** — "reified 입력을 StarE에 넣는다"가 한 가지로 정해지지 않는다(§0-1, §4-2). Difficulty **Complex** —
외부 코드 1개(5,400줄), 새 의존성 1개, GPU 장시간 실행, 재개 로직, 공급망 검토.

## 0. 사용자 결정 (미확정)

| # | 결정 | 권장 | 대안과 대가 |
|---|---|---|---|
| 0-1 | **reified arm을 StarE 안에서 어떻게 만드나** | **(가) 그래프 쪽 reification.** 인코더(메시지 전달)가 보는 그래프만 "문장 노드(blank) + 역할 엣지"로 푼다. 디코더(Transformer)가 받는 질의 토큰 `[s, r, (q, v)…]`는 두 arm이 **같다.** 5장의 `reify()` 계약을 그대로 옮긴 것이고(blank는 관계 이름만 가진다), StarE 원본 파일을 고치지 않고 서브클래스 하나로 된다(§4-2) | **(나) (가) + 문장 노드마다 학습 임베딩**: 대조군에 파라미터 약 4.5M을 더 준다(steelman). 파일럿이 끝난 뒤 추가 arm으로 돌릴 수 있다. **(다) 질의까지 reification**(test 문장의 blank 노드로 질의): StarE 프로토콜에서는 test 문장이 그래프에 없으므로 그 노드의 임베딩이 존재하지 않는다. **불가능**하며 넣지 않는다 |
| 0-2 | **파일럿의 split** | **(가) 그래프 = train, 평가 = valid**(`USE_TEST=False`). test는 손대지 않고 판정 실행(§7)을 위해 남긴다 | (나) StarE 기본값(그래프 = train+valid, 평가 = test): 논문 수치와 같은 설정이지만 파일럿이 test를 소모한다 |
| 0-3 | **파일럿 예산** | **(가) 스모크 3 epoch로 epoch당 시간을 재고, arm당 45분 이하가 되는 epoch 수를 §5-b에 적은 뒤 돌린다**(추정 약 100 epoch, §5). dev seed 0 하나, 두 arm 순차 | (나) StarE 기본 401 epoch: arm당 약 2~3시간(추정). "작은 규모"가 아니다. 판정 실행(§7)의 예산 후보로 남긴다 |
| 0-4 | **StarE 코드와 의존성을 어디에 두나** | **(가) `third_party/StarE/`에 commit 고정 clone(git 무시), 추적 파일은 `third_party/stare.lock.json`(URL·commit·sha256)과 `.harness/scripts/fetch_stare.py`.** `torch-geometric`는 `pyproject.toml`의 선택 extra `stare`로 잠근다. `torch_scatter`·`wandb`는 설치하지 않고 저장소 안의 shim으로 대신한다(§3-2) | (나) 저장소 밖 형제 폴더: 재현 경로가 저장소 밖으로 샌다. (다) git submodule: Windows·`--frozen` 흐름에 마찰이 크다 |
| 0-5 | **CD 정지 해제** | 이 작업에 한해 해제. 해제 없이는 §9의 어떤 파일도 만들지 않는다 | 유지: charter만 남긴다 |

## 1. 리뷰가 지적한 "0.147 vs 0.654"는 왜 비교가 안 되는가 — 정확한 차이

5장 §5.11은 "과제 정의와 채점이 달라 직접 비교할 수 없다"고만 적었다. 리뷰는 그 설명만으로는 방어가 어렵다고 봤다.
아래는 두 프로토콜의 차이를 코드에서 읽어 센 것이다. 이 표 자체가 논문 5장에 들어가야 할 내용이다.

| 항목 | 5장 V2-D (우리 인코더) | StarE 논문·코드 (`run.py` 기본값) |
|---|---|---|
| 가리는 위치 | 문장의 **모든 역할 중 하나를 무작위로**(주어·목적어·한정값 전부 후보) | **주어 또는 목적어만**(`CORRUPTION_POSITIONS [0, 2]`). 한정값은 절대 예측하지 않는다 |
| 후보 집합 | 엔티티 18,791개 전부 | 주어·목적어 자리에 한 번이라도 나온 10,874개. 한정값으로만 나온 7,917개는 후보에서 뺀다(`ENT_POS_FILTERED`) |
| 학습 예산 | 1,500 step × batch 32 = 48,000 문장-갱신, d = 64, 2층 | 401 epoch × 43,716 키 ≈ 1,750만 행-갱신, d = 200, GCN 2층 + Transformer 2층 |
| 손실 | 보완 질의 vs 후보의 대조 손실 | 전체 엔티티에 대한 1-vs-all BCE, 라벨 스무딩 0.1, 역방향 문장 추가 |
| 평가 표본 | test 5,297 중 **2,000개 표본**, forward 하나가 128개를 동시에 가림 | test 5,297개 **전부** × 2방향 = 10,594 질의, 각각 따로 채점 |
| 평가 시 그래프 | 전체 그래프(held-out 포함, 가린 멤버만 제외) | train+valid 문장만. test 문장은 그래프에 없다 |
| 한정쌍 처리 | 한 문장 안의 **같은 속성 반복은 버린다**(4,660쌍 제외, 평균 항수 3.30, 최대 7) | 반복을 **유지**하고 한 문장당 **6쌍까지만** 받는다(`MAX_QPAIRS 15`; train 55문장이 잘린다) |

같은 데이터지만 **다른 과제, 다른 후보, 다른 예산, 다른 표본**이다. 그래서 이 charter는 우리 수치를 StarE 수치에 맞추려 하지
않고, **StarE의 과제·후보·예산·표본 그대로** 두 입력 표현을 넣는다. 그러면 StarE 논문의 0.654(hyper arm 재현 여부)가
그대로 참조점이 된다.

## 2. 사실 — WD50K_100 (정확한 수, 세는 법 포함)

`data/wd50k_100/{train,valid,test}.txt`. StarE 저장소 `data/clean/wd50k_100/statements/`와 sha256이 같다(3개 파일 모두).

| 항목 | 수 | 세는 법 |
|---|---|---|
| 문장 | 31,314 = 22,738 / 3,279 / 5,297 | 파일 줄 수(train / valid / test) |
| 엔티티(모든 자리) | 18,791 | 주어·목적어·한정값 자리의 Q-id 합집합. StarE `n_entities` 18,792는 패딩 id 0을 포함 |
| 주어·목적어 자리 엔티티 | 10,874 | 후보 집합(§1) |
| 한정값으로만 나온 엔티티 | 7,917 | 18,791 − 10,874. StarE README는 7,862라 적었다(세는 법 차이, 코드가 세는 값은 7,918 = 7,917 + 패딩) |
| 주술어 | 204 | 두 번째 자리 P-id |
| 한정 속성 | 134 (주술어와 60개 겹침) | 짝수 자리(4번째부터) P-id. 합집합 278; StarE `n_relations` 279는 패딩 포함. **StarE는 주술어와 한정 속성을 한 id 공간에 둔다** |
| 한정쌍 | 45,472 (전체), 같은 속성 반복 4,660 | 문장당 `(속성, 값)` 쌍 |
| 항수 | 평균 3.45, 분포 3:20,426 / 4:9,342 / 5:876 / 6:306 / 7:256 / 8+:138 | 2 + 한정쌍 수(반복 포함). 5장의 3.30은 반복을 버린 값 |
| train에 없는 엔티티 | valid 1,107 / test 253 | 0-2 (가)에서는 valid 엔티티 1,107개가 그래프 엣지 없이(자기 루프만) 평가된다 — 두 arm에 똑같이 적용 |
| 완전 중복 문장 | 68 | 전체 파일에서 같은 줄 |

**StarE가 실제로 쓰는 그래프(0-2 (가), train만)**: 문장 22,738 → 엣지 45,476(역방향 포함), 한정쌍 32,439(6쌍 절단 후, 역방향 복제 전).
**reified 그래프(§4-2)**: 문장 노드 22,738, 역할 엣지 77,915(= Σ(2 + 유지된 한정쌍 수)), 역방향 포함 155,830.
0-2 (나)면 문장 노드 26,017, 역할 엣지 89,208.

**학습 1 epoch**: 0-2 (가)에서 sampler 키(고유 `(s, r, quals)` + 역방향) 38,636개 → batch 128로 301 step.
평가 질의: valid 6,558개(3,279 × 2방향), test 10,594개.

## 3. StarE 코드 — 고정 사실과 실행 가능성

### 3-1. 원본

- `https://github.com/migalkin/StarE`, commit **`b294b9e2cde97ab9e81d2736b7f1f0959c9fd98b`**(2023-12-01, 저장소 최신), MIT.
- 요구: `torch==2.1.1`, `torch-scatter==2.1.2`, `torch_geometric==2.4.0`, `tqdm`, `wandb`. 이 저장소 환경은 Python 3.13, torch 2.10+cu130.
- 모델: `stare_transformer` = StarE(H) 인코더(CompGCN 계열 메시지 전달 + 한정쌍 집계 `φ_q`, `W_q`) + Transformer(H) 디코더.
  GCN 층은 코드상 **최대 2층**(`conv1`, `conv2`만 존재).
- 프로토콜(§1 오른쪽 열). 지표는 MRR·MR·Hits@1/3/5/10을 **목적어 예측 절반과 주어 예측 절반의 평균**으로 낸다.
- 결정성: `run.py`가 numpy 42 / random 42 / torch 132로 고정. 엔티티·관계 id는 정렬된 집합에서 매기므로 실행 간 같다.
- 저장·재개: **없다.** 학습 루프는 epoch마다 평가만 하고 가중치를 저장하지 않는다.

### 3-2. 이 환경에서 돌아가는가 — 2026-10-08 CPU 탐침 (scratchpad, 저장소 밖, 결과 아님)

- `torch-geometric 2.6.1`은 torch 2.10 위에서 import 된다(`uv run --with` 임시 환경, lock 미변경).
- `torch_scatter`는 torch 2.10 / cu130 / Windows용 컴파일 wheel이 없다. StarE가 쓰는 함수는 `scatter_add`·`scatter_mean`·`scatter_max` 셋뿐이고
  모두 torch 기본 연산(`index_add_`, `scatter_reduce`)으로 30줄에 대체된다. `wandb`는 `WANDB=False`라도 `import`가 필요하므로 빈 stub를 둔다.
- 이 shim 둘만으로 **원본 파일을 한 줄도 고치지 않고** `StarE_Transformer`가 WD50K_100 위에서 구성·순전파·역전파됐다:
  파라미터 **5,068,016**(기본 d = 200), batch 128 한 step CPU 2.39초, loss 0.959.
- 따라서 GPU epoch당 시간은 **아직 재지 않았다.** §5의 시간은 추정이며 스모크에서 실측한다.

## 4. 설계 — 같은 백본, 다른 입력 표현

### 4-1. `hyper` arm — StarE 원본 그대로

`run.py`의 기본 설정(`stare_transformer`, d 200, batch 128, lr 1e-4, Adam, StepLR(50, 0.95), 라벨 스무딩 0.1, grad clip 1.0,
`QUAL_REPR sparse`, `TRIPLE_QUAL_WEIGHT 0.8`, `QUAL_AGGREGATE sum`, `OPN rotate`)을 **바꾸지 않는다.** 데이터도 `DataManager.load`가 주는 그대로다.
우리가 추가하는 것은 (1) seed 인자, (2) 10 epoch 단위 체크포인트·재개, (3) 지표를 `metrics.jsonl`에 덧붙여 쓰는 것뿐이며,
이 셋은 두 arm에 똑같이 적용된다.

### 4-2. `reified` arm — 그래프 쪽 reification (0-1 (가))

5장의 `Hypergraph.reify()` 계약을 StarE의 자료형으로 옮긴다. 그래프 split G(0-2 (가)면 train)의 문장 i = `(s, p, o, (q₁, v₁), …)`에 대해:

- **문장 노드** `b_i = E + i`(E = 실제 엔티티 수 18,792, 패딩 포함). 문장마다 하나.
- **역할 엣지**(StarE `edge_index` 열, `edge_type`): `b_i → s` 타입 `role:subject`(= R), `b_i → o` 타입 `role:object`(= R + 1),
  유지된 한정쌍마다 `b_i → v_k` 타입 **`q_k` 그대로**(한정 속성의 관계 id를 재사용). StarE가 역방향 엣지를 자동으로 더하므로 양방향이 된다.
  반복 속성은 유지하고 6쌍 절단은 hyper arm과 같게 둔다(§1).
- **관계 표**: 기존 R = 279 + 새 역할 관계 2 = 281(역방향 포함 562행). 주술어는 관계 표에 그대로 있고 디코더가 쓴다.
- **문장 노드의 초기 임베딩**: `x[b_i] = init_rel[p_i] + w_blank`. 즉 blank는 **자기 주술어의 관계 임베딩 + 학습되는 공통 벡터 하나**로
  시작한다. 5장 `reify()`에서 blank의 text가 관계 이름뿐인 것과 같다(classic reification의 `rdf:type` 트리플). 문장별 학습 임베딩은 **없다**(0-1 (나)가 그것이다).
- **인코더**: StarE 자체의 **triple 모드**(`STATEMENT_LEN 3`) `StarEConvLayer` 2층. 한정쌍 집계(`W_q`)가 없는 CompGCN이다.
  두 arm의 층 수·차원·dropout·정규화·활성은 같다.
- **디코더**: `StarE_Transformer`의 디코더 그대로. 질의 토큰 `[s, r, q₁, v₁, …]`은 hyper arm과 **같은 텐서**다. 엔티티 토큰은 `x`의 실제 엔티티 행을,
  관계 토큰은 관계 표를 읽는다.
- **후보 채점**: `x[:E]`만 후보다. 문장 노드는 학습·평가 어디서도 후보가 아니다.
- **그 밖 전부 동일**: sampler, 손실, 라벨 스무딩, 역방향 문장, 최적화기, 스케줄, grad clip, 평가 필터링, 후보 제외 집합, 위치 `[0, 2]`.

구현은 `StarE_Transformer`를 **상속한 서브클래스 하나**가 `forward_base`를 덮어써 `x = cat(init_embed, x_blank)`를 만들고 후보를 `x[:E]`로 자르는 것이다.
원본 파일은 고치지 않는다.

### 4-3. 같게 할 수 없어서 적어 두는 것

| 항목 | hyper | reified | 비고 |
|---|---|---|---|
| 파라미터 | 5,068,016(실측) | 약 4,989,000(**구현 후 실측해 적는다**) | reified에 `W_q` 2층 × 200×200 = 80,000이 없고, 관계 2개(×2 방향) 800 + `w_blank` 200이 있다. **대조군이 약 1.6% 작다** |
| 유효 깊이 | 문장 참여자가 한 층에서 만남 | blank를 거쳐 두 층 | 5장 §5.7의 깊이 반박. StarE가 2층 고정이라 **4층 steelman은 이 백본에서 불가** — 한계로 적는다 |
| 그래프 크기 | 엣지 45,476 + 한정쌍 32,439 | 엣지 155,830 | 5장 §5.9와 같은 방향(reified가 더 큼). VRAM·시간을 함께 기록 |

### 4-4. 왜 (다)가 아닌가

StarE는 test 문장을 그래프에 넣지 않는다. reified 표현에서 test 문장의 blank 노드는 그래프에 없으므로 임베딩이 없고, 그 노드로 질의할 수 없다.
5장은 평가 시 전체 그래프를 쓰는 다른 프로토콜이라 가능했다. 이 차이는 reification의 **알려진 약점**(새 문장의 노드가 없다)이지 구현 한계가 아니며,
그래서 (가)는 "그래프 표현만 다르고 질의는 같다"로 질문을 좁힌 것이다.

## 5. 파일럿 프로토콜 (판정 아님)

1. **스모크** — 두 arm 각 3 epoch, dev seed 0, 0-2 (가). 목적: 파이프라인이 끝까지 돌고, 재개가 동작하고, epoch당 시간과 peak VRAM이 나오는 것.
2. **예산 고정(§5-b 추가 기록)** — `E_pilot = min(401, floor(45분 / epoch당 시간))`, 하한 50. 두 arm 중 **느린 쪽** 기준. 평가는 10 epoch마다 valid.
   이 값을 파일럿 결과를 보기 전에 이 charter에 적는다.
3. **파일럿 실행** — hyper arm 끝난 뒤 reified arm(순차, 동시 실행 금지). 10 epoch 단위 체크포인트(모델·옵티마이저·스케줄러·epoch), 리셋 시 `--resume`.
4. **보고** — arm별 마지막 평가와 최고 평가의 valid MRR / Hits@1 / Hits@3 / Hits@10(StarE 방식: 두 방향 평균), epoch별 곡선, 파라미터 수, wall, peak VRAM.
   `docs/stare-input-representation-v1-results.md`.

추정 시간(실측 전): CPU step 2.39초를 GPU에서 약 0.05~0.1초로 보면 epoch(301 step) 15~30초 + 평가 → 100 epoch ≈ 30~50분/arm.
틀릴 수 있으므로 스모크 뒤 §5-b로 대체한다.

**이 파일럿이 답하지 않는 것**: seed 1개, 줄인 예산이므로 어떤 "우위"도 판정이 아니다. 5장 V2-D의 verdict를 바꾸지도 않는다.

### 5-b. 예산 동결 (2026-10-08, 스모크 뒤·파일럿 결과 관측 전)

스모크(`outputs/stare_input_representation_v1/smoke/{arm}/seed_0/`, dev seed 0, 0-2 (가), 2 epoch 뒤 `--resume`으로 3 epoch) 실측:

| arm | 파라미터 | 그래프(역방향 포함) | epoch당 학습(3 epoch 평균) | valid 평가 1회(6,558 질의) | peak VRAM |
|---|---|---|---|---|---|
| hyper | 5,068,016 | 엣지 45,476 + 한정쌍 64,878 | **35.4 s**(35.7 / 34.9 / 35.5) | 1.8 s | 948 MB |
| reified | 4,989,016 (§4-3 산식과 일치: −80,000 + 800 + 200) | 엣지 155,830, 노드 41,530 | 29.0 s(28.9 / 28.4 / 29.7) | 2.4 s | 1,386 MB |

- **느린 쪽은 hyper**(35.4 s). 한정쌍 집계 `W_q`가 있어 reified의 triple 모드보다 느리다. 그래프가 3.4배 큰 reified가 더 빠른 것은 예상 밖이었고 그대로 적는다.
- **E_pilot = min(401, floor(2,700 s / 35.4 s)) = 76.** 하한 50 이상이므로 그대로 쓴다. 평가는 10 epoch마다와 마지막(76)에서 valid, 체크포인트 10 epoch 단위.
- 예상 시간: hyper 76 × 35.4 s + 8 × 1.8 s ≈ **45분**, reified ≈ 37분. 순차 실행, 총 약 1시간 25분.
- 재개 경로는 두 arm 모두 GPU에서 확인했다. 첫 hyper 재개는 `RNG state must be a torch.ByteTensor`로 실패했다 — 체크포인트를 `map_location=cuda`로 읽어 RNG 텐서가 GPU에 올라간 탓이며, 항상 CPU로 읽고 `.cpu()`를 강제하도록 고친 뒤(테스트 `test_restore_accepts_rng_states_from_any_device` 추가, 수정 제거 시 실패 확인) 재실행해 통과했다. 판정 규칙·예산 산식은 바뀌지 않았다.
- 스모크의 valid 수치(hyper 3 epoch MRR 0.0117, reified 3 epoch 참조)는 **예산 결정에 쓰지 않았고 결과로 읽지 않는다.**

### 5-c. 운영 기록 — 파일럿 중 기계 리셋 2회 (2026-10-08, 예산·규칙 불변)

| 시각 | 사건 | 상태 |
|---|---|---|
| 02:52 → ~03:11 | hyper 파일럿 시작, epoch 31 진행 중 **Kernel-Power 41 하드 리셋**(GPU 82 °C, 76% util, 2.7 GB, 전력 한도 기본 200 W) | 체크포인트 epoch 30 온전. `metrics.jsonl` 꼬리 148바이트가 0으로 채워짐(저장 시 fsync 없었음) |
| 03:32 → ~03:37 | epoch 30에서 재개(사용자가 `nvidia-smi -pl 160` 적용), epoch 39 직후 **두 번째 리셋** | 체크포인트 epoch 35 온전(fsync 적용 후 NUL 0바이트) |
| 03:50 → 04:41 | epoch 35에서 재개, step당 150 ms GPU 휴지(듀티 약 45%), 체크포인트 5 epoch, 사용자가 `nvidia-smi -lgc 300,1400` 적용 → 1,395 MHz · 52~54 W · 63 °C | hyper 76 epoch 완료, 리셋 없음 |
| 04:42 → 06:18 | reified 76 epoch, 같은 휴지·클럭 설정 | 완료, 리셋 없음 |

결과: `docs/stare-input-representation-v1-results.md`. **사전 예측(§6)은 틀렸다** — 76 epoch에서 reified가 앞섰다(MRR 0.434 vs 0.336). §6의 규칙대로 기록했고 §7로 넘긴다.

- VRAM은 원인에서 제외(peak 948 MB / 8 GB). 사용자 판단은 순간 전력 피크이며, 전력 한도만으로는 두 번째 리셋을 막지 못했고 클럭 상한을 추가했다.
- 러너 변경(결과 불변): 체크포인트·지표 저장에 fsync, 재개 시 체크포인트 이후 행 자동 정리(`prune_jsonl`), NaN/inf 체크포인트 재개 거부, 기존 지표 위에 `--resume` 없는 시작 거부, `--step-sleep-ms`(옵티마이저 step 사이 GPU 휴지). 테스트 22개.
- **해석에 미치는 영향**: 학습 수학은 동일하다(같은 seed·같은 step 순서를 체크포인트가 그대로 잇는다. 단, CUDA 비결정 연산 때문에 리셋 없이 한 번에 돌린 것과 비트 단위로 같지는 않다). `epoch_time_s`에는 휴지가 포함되므로 §5-b의 epoch당 시간과 **비교할 수 없다.** 결과 문서는 `sleep_s`를 뺀 학습 시간을 따로 적는다.
- reified 파일럿도 같은 휴지·클럭 설정으로 돌린다. 세 번째 리셋이 나면 멈추고 하드웨어 판단으로 넘긴다.

## 6. 사전 예측 (틀릴 수 있게 적는다)

- hyper arm의 valid MRR이 E_pilot에서 reified arm보다 높을 것이다. 차이는 MRR 0.02~0.06 범위로 본다. 근거: StarE 논문에서 한정쌍을 그래프에서
  빼면(StarE(T)) 성능이 떨어졌고, reified는 한정쌍을 **남기되 두 층이 걸리는** 중간 표현이다.
- hyper arm 100 epoch의 valid MRR은 StarE 논문의 test 0.654보다 낮을 것이다(예산 1/4, valid에 train 밖 엔티티 1,107개).
- **읽기 규칙(미리 고정)**: seed 1개에서 MRR 차이 < 0.01은 "구별 안 됨"으로 읽는다. reified ≥ hyper면 버그를 찾는 대신 그대로 적고,
  §7의 5 seed 판정으로 넘긴다 — 그 경우 5장의 주장은 백본 의존으로 **한정**될 후보가 된다.

## 7. 후속 판정 실행 (이 charter 밖, 미리 적어 둠)

파일럿이 끝나면 **별도 addendum**으로 동결한다: seed `1 2 3 4 5`, 예산은 E_pilot 또는 401 epoch 중 파일럿 곡선이 보여 준 수렴 지점,
split은 StarE 기본(0-2 (나))으로 논문 수치와 같은 자리에 놓을지 결정, 판정 규칙은 V2-D와 동일(**MRR과 Hits@10 모두 ≥ 4/5 seed**, 동점은 우위 아님).
JF17K 또는 WikiPeople 추가(리뷰 ①의 두 번째 데이터셋)는 그 다음이다.

### 7-a. 판정 실행 동결 (2026-10-09, 사용자 지시 "5 seed × 401 epoch × 2 arm으로 판정 실행해", 결과 관측 전)

파일럿(§5-b·`docs/stare-input-representation-v1-results.md`)은 사전 예측과 반대였고 곡선이 교차했다. 그래서 판정은 **양측**이다 — 방향을 미리 정하지 않는다.

**고정 조건**

| 항목 | 값 |
|---|---|
| seed | `1 2 3 4 5` (파일럿의 dev seed 0은 쓰지 않는다) |
| 예산 | StarE 기본값 그대로: **401 epoch**, batch 128, Adam lr 1e-4, StepLR(50, 0.95), 라벨 스무딩 0.1, grad clip 1.0 |
| split | StarE 기본(`USE_TEST=True`): 그래프 = train + valid 26,017문장, 평가 = **test 10,594 질의**(5,297 × 2방향), filtered 인덱스 = train + valid + test. 논문 수치(0.654)와 같은 자리 |
| 평가 시점 | 10 epoch마다와 **401**에서 test. 판정은 **마지막 epoch(401)의 값**이다. test에서 고른 최고값은 낙관적 참고로만 적고 판정에 쓰지 않는다 |
| 실행 순서 | seed 1의 hyper → reified, seed 2의 hyper → reified, … (리셋이 와도 짝이 맞는 seed부터 남게) |
| 운영 | §5-c 설정 그대로: step당 150 ms GPU 휴지, 체크포인트 5 epoch, 사용자 클럭 상한 1,400 MHz·전력 160 W. 드라이버 `.harness/scripts/run_stare_judged.ps1`가 10개를 순차 실행하고, 리셋 뒤 같은 명령이 완료 실행은 건너뛰고 부분 실행은 체크포인트에서 잇는다 |
| 산출물 | `outputs/stare_input_representation_v1/judged/{arm}/seed_{k}/`, 판정 `judged/judgment.json`·`summary.csv` (`.harness/scripts/judge_stare_input_representation.py`) |

**판정 규칙** (V2-D §7과 같은 강도, 양측)

- 지표: test **MRR**과 **Hits@10**, 401 epoch.
- seed마다 값이 **엄격히** 큰 arm이 그 seed를 이긴다. 동점은 누구의 승리도 아니다.
- `reified_wins`: reified가 두 지표 모두 ≥ 4/5 seed. `hyper_wins`: 대칭. 그 밖: `no_consistent_order`.
- 보고(판정 아님): seed별 표, Hits@1·Hits@3·MR, 쌍 차이의 평균·95% 구간(t, 자유도 4)·양측 t p·정확 부호 뒤집기 p(5 seed 최솟값 0.0625), 목적어/주어 예측 분리, 곡선.

**이식 건전성 확인**(판정 아님): hyper arm은 StarE 원본을 기본 예산으로 돌린 것이므로 5 seed 평균 test MRR이 논문의 0.654에서 **±0.03 안**이어야 한다. 벗어나면 결과를 해석하기 전에 이식(shim·torch 2.10·데이터 로더)을 먼저 점검한다. 이 확인은 판정을 바꾸지 않고 "판정을 믿을 수 있는가"만 가른다.

**사전 예측**(틀릴 수 있게): 파일럿의 교차 때문에 자신 없다. 굳이 적으면 `reified_wins` 55%, `no_consistent_order` 30%, `hyper_wins` 15%. 이식 확인은 통과할 것으로 본다(차이 0.02 안).

**읽기 규칙**: 어느 verdict든 결과다. `reified_wins`면 5장의 주장은 "우리 인코더에서의 관찰"로 한정하고 논문에 StarE 결과를 한계 아닌 **결과**로 싣는다. `hyper_wins`면 파일럿의 역전은 예산 효과였다고 적는다. `no_consistent_order`면 백본 간 일반화를 주장하지 않는다. 어느 경우도 arm·예산·규칙을 결과를 본 뒤 바꾸지 않으며, 세 번째 arm(대칭 집계기)은 별도 charter다.

**비용**: 실행당 401 × (학습 37 s + 휴지 45 s) ≈ 9.1 h(hyper), 401 × (30 + 45) ≈ 8.4 h(reified) → 10개 ≈ **88시간(3.7일)**, 평가 41회 × 3 s는 무시. 사용자가 이 시간을 승인했다. Windows Update 일시 중지는 2026-10-20까지다.

**결과 (2026-10-13 11:57, 실제 약 104시간, 리셋 0회)**: verdict **`hyper_wins`** — hyper가 MRR·Hits@10 모두 5/5 (쌍 차이 MRR −0.0068 [−0.0104, −0.0031], Hits@10 −0.0058 [−0.0073, −0.0043]).
이식 확인 통과(hyper 평균 test MRR 0.6553 vs 0.654). 사전 예측의 15% 갈래. 실제 소요는 train+valid 그래프의 step 수(342)가 추정(301)보다 많아 88시간이 아니라 104시간이었다.
결과 문서 `docs/stare-input-representation-v1-results.md`. 규칙·예산·arm은 결과를 본 뒤 바꾸지 않았다.

## 8. Non-goals

- SOTA 재현·갱신 없음. StarE 하이퍼파라미터 탐색 없음(기본값 고정).
- 두 번째 데이터셋(JF17K·WikiPeople), 두 번째 백본(HINGE·GRAN·QUAD·HAHE 등) 없음 — 후속.
- 5장 verdict·수치 수정 없음. 우리 인코더(`src/ontology/encoder.py` 계열) 변경 없음.
- LLM 없음, 센서 데이터 없음, 리뷰 ②③④ 없음(§13).
- StarE 원본 파일 패치 없음. 적응은 shim 모듈과 서브클래스로만 한다.
- 질의까지 reification(0-1 (다)) 없음.

## 9. Touched files (0-5 해제 뒤에만)

- 신규 `third_party/stare.lock.json`(URL·commit·라이선스·데이터 sha256), `third_party/README.md`; `third_party/StarE/`는 clone 대상(**git 무시**).
- 신규 `.harness/scripts/fetch_stare.py` — 고정 commit clone, commit·sha256 검증. **네트워크를 쓰는 유일한 단계.**
- 신규 `src/ontology/stare_bridge/` — `shims/torch_scatter/__init__.py`, `shims/wandb/__init__.py`, `reify_stare.py`(정수 문장 → reified `graph_repr`, 순수 함수),
  `model.py`(`ReifiedStarE_Transformer` 서브클래스), `runner.py`(`run.py`의 오케스트레이션 재현 + seed + 10 epoch 청크 재개 + `metrics.jsonl` + `LONG_RUN.lock`).
- 신규 `tests/test_stare_bridge.py` — reified 변환의 개수 불변식(문장 노드 수 = 문장 수, 역할 엣지 수 = Σ(2 + 유지 한정쌍), 관계 id 재사용, 후보 절단),
  shim과 단순 반복문의 동치, 재개 상태 왕복. `torch_geometric`이 없으면 해당 테스트는 skip.
- 신규 `.harness/scripts/launch_stare_arm.ps1` — `Start-Process` 분리 실행(기존 `launch_hyperedge_arm.ps1` 패턴).
- 수정 `pyproject.toml`·`uv.lock`(`torch-geometric>=2.6,<2.7`; **2026-10-08 구현 시 extra가 아니라 일반 의존성으로 넣었다** — `uv sync --frozen`은 exact sync라 extra를 지워 버리며, `sentence-transformers`를 일반 의존성으로 둔 기존 근거와 같다. lock은 기존 `exclude-newer` 2026-08-13을 그대로 써서 추가 10개 외 어떤 pin도 바꾸지 않았다), `requirements.txt`(export 결과만),
  `.gitignore`(`third_party/StarE/`, `outputs/stare_input_representation_v1/**/*.pt`), `.harness/checkpoints/default.yml`(advisory `stare_bridge_contract`).
- 수정 `docs/charter.md`(addendum), `docs/checkpoints.md`, 이벤트 로그. 결과 `docs/stare-input-representation-v1-results.md`.
- 산출물 `outputs/stare_input_representation_v1/{hyper,reified}/seed_0/{metrics.jsonl,summary.json,run_config.json}`.

## 10. 안전·운영

- **GPU 운영**(메모리 규칙): 이 기계는 GPU 부하가 길면 하드 리셋된다. 실행은 분리 프로세스, 덧붙여 쓰는 로그, 10 epoch 청크 재개. 실행 중 `.harness/tmp/LONG_RUN.lock`을 두고
  그동안 다른 python/pytest/uv를 돌리지 않는다. 시작 전 사용자에게 Windows Update 일시 중지를 요청한다. 두 arm은 순차.
- **공급망**: `torch-geometric`는 순수 Python이고 버전을 `<2.7`로 잠근다. `pip-audit` 게이트를 통과해야 한다. `wandb` stub는 네트워크·텔레메트리가 없다.
  StarE는 commit 고정 + 데이터 sha256 검증. `security-reviewer`에게 `pyproject.toml`·`uv.lock` diff와 `fetch_stare.py`를 넘긴다.
- **원본 무결성**: `fetch_stare.py`가 clone 뒤 `git rev-parse HEAD`와 lock 파일의 commit 일치를 확인한다. 테스트는 shim 경로가 원본 import를 가로채는 것만 확인하고
  원본 파일 내용은 건드리지 않는다.

## 11. Required checkpoints

`charter_draft` → `user_decisions`/`cd_halt_lifted` → `stare_fetched`(commit·sha256 일치) → `deps_locked`(`uv lock --exclude-newer`, `pip-audit`, `uv sync --frozen`) →
`stare_bridge_contract`(테스트 + Ruff) → `smoke_both_arms`(3 epoch × 2, 재개 1회 강제 확인) → `budget_addendum`(§5-b) → `pilot_hyper` → `pilot_reified` →
`results_doc` → `checkpoint_gate`(전체 테스트 + lint) → `clarification_debt`.

## 12. Rollback

신규 파일 전부 삭제(§9), `pyproject.toml`·`uv.lock`·`requirements.txt`·`.gitignore`·`default.yml`의 변경 되돌림, `third_party/StarE/`와
`outputs/stare_input_representation_v1/` 제거. `src/ontology`의 기존 모듈은 바뀌지 않으므로 5장 산출물과 판정은 영향받지 않는다.

## 13. 리뷰의 다른 항목 — 이 charter 범위 밖 (기록만)

- **② LLM 고정**(7B급 오픈 모델 4bit 로컬, temperature 0으로 8장 재측정): 별도 charter. 8GB VRAM에서 7B 4bit는 올라가지만 이 파일럿과 동시에 돌릴 수 없다.
- **③ RAG 대조군 강화**(같은 사실을 텍스트로 온전히 주는 arm, 코드 집계로 안 풀리는 과제): 별도 charter. `docs/pipeline-vs-rag-v2-results.md`의 닫힌 방향과 겹치는 부분을 먼저 가려야 한다.
- **④ 산업 도메인**: 데이터 이용 허락 또는 공개 제조 데이터셋 선정이 선행. 실험 charter 이전의 결정이다.
- **1·4·5 논문 구성·초록·키워드**: 실험이 아니라 집필 작업. 이 charter의 결과가 나온 뒤 5장을 다시 쓸 때 §1의 표를 넣는다.
