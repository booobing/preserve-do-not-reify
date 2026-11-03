# HNN 온톨로지 Encoder v2 — 네 개 통제 결과 (Paper 1, 2026-08-22)

판정 규칙·예산·seed는 `docs/charter-hnn-ontology-encoder-v2.md` §7과 §7-b에 **결과 관측 전** 동결됐다. 이 문서는
`outputs/hnn_ontology_encoder_v2/{factor}/judgment.json`·`summary.json`의 수치를 옮긴 것이며, 규칙을 결과를 본 뒤에 바꾸지 않았다.
v1 결과는 `docs/hnn-ontology-encoder-v1-results.md`.

## 이 문서에서 쓰는 용어

- **하이퍼엣지(`hyper` arm)**: "누가·무엇을·어떤 조건에서"처럼 참여자가 셋 이상인 사실 하나를, 참여자 묶음 하나로 통째로 인코딩하는 방식.
  제안 방식이다.
- **이진 reification(`reified_binary` arm)**: 같은 사실을 "사실 그 자체를 가리키는 빈 노드(blank node) + 참여자마다 2항 연결"로 풀어 쓰는,
  RDF·Wikidata가 실제로 쓰는 표준 표현. 정보량은 같고 표현만 다르므로 판정 대조군이다.
- **`flat_text` arm**: 구조를 전혀 쓰지 않고 문장만 인코딩하는 참조선. 판정에는 들어가지 않고, "이 과제가 구조 없이 어디까지 풀리는지"를
  보여주기 위해 함께 보고한다.
- **retrieval recall@5**: 사실을 묘사한 질의를 넣었을 때 정답 사실이 상위 5위 안에 든 비율. 검색이 되는지를 본다.
- **completion acc@1 / Hits@10 / MRR**: 사실의 참여자 하나를 가리고 후보 중 정답 순위를 매긴 결과. acc@1은 1위 적중률, Hits@10은 10위 안
  적중률, MRR은 정답 순위의 역수 평균(1위=1.0, 5위=0.2)이다. **구조가 필요한 과제**이며, 이 프로그램의 핵심 지표다.
- **transductive / inductive**: transductive는 평가할 사실의 **구조가 학습 그래프에 이미 있는** 설정이고, inductive는 그 구조를 학습에서
  빼는 설정이다. 실제 온톨로지는 새 사실이 계속 들어오므로 inductive가 현실에 가깝다. v1은 transductive였다.
- **헤드 정합(head parity)**: 두 arm이 사실 벡터를 꺼내는 출력층을 같은 것(`edge_out`)으로 맞추는 것. v1에서는 `reified_binary`만 다른
  출력층을 써서, 격차의 일부가 메커니즘이 아니라 이 비대칭 탓일 수 있다는 지적이 있었다. **대조군을 강화하는 수리**다.
- **동점 비관 처리 / filtered ranking**: 점수가 같으면 정답을 뒤로 미뤄 제안에 유리하지 않게 세고, 학습에 이미 있는 다른 정답은 후보에서 빼서
  "맞았는데 틀렸다"로 세지 않는 표준 처리.

## 한 줄 결론

v1의 결론("N항 사실을 하이퍼엣지로 보존하면 같은 정보를 이진 reification으로 푼 것보다 낫다")은 **네 개의 통제를 각각 통과했다.**
inductive 프로토콜, reified 헤드 정합, 강한 텍스트 타워, 실데이터(WD50K_100) 어디에서도 `hyper`가 `reified_binary`를 사전등록 지표
두 개 모두 **5/5 seed**로 이겼고(네 factor 전부 verdict `success`), 넷을 동시에 켠 결합 조건에서도 순서가 뒤집히지 않았다.
다만 **절대 성능은 통제를 걸수록 크게 떨어졌고**, 검색은 구조 없이도 되는 과제라는 v1의 해석은 강한 텍스트 타워에서 더 뚜렷해졌다.

## 공통 고정값

- 두 arm은 **같은 인코더 모듈·같은 Vocab·같은 파라미터 수·같은 seed의 동일 초기 가중치·같은 학습 예산·같은 텍스트 타워**를 쓴다.
  이 공정성 조건은 v1에서 테스트로 고정했고 v2 신규 코드가 깨지 않는지 `tests/test_ontology_v2.py`에서 다시 검사한다.
- 예산: 1,500 step, batch 32, AdamW lr 2e-3, d=64, 2층. 판정 seed `1 2 3 4 5`. deterministic 모드 기본
  (`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`). 장치 RTX 3060 Ti, 환경 `uv run --frozen`.
- 합성 월드 `v1`: 엔티티 78개, 사실 390개, split train/val/test = 272/59/59(사실 단위). 검색 질의 117개, completion 118개로 채점.
- 판정: `hyper` vs `reified_binary`, 사전등록 지표 두 개 모두 **≥4/5 seed** 우위여야 `success`. 동점은 우위로 세지 않는다.
- 각 factor는 v1 설정에서 **딱 한 가지만** 바꾼다(결합 조건 제외). 이는 러너의 factor 정의와 테스트로 강제된다.

## Factor별 결과

### v1_baseline — v2 하네스에서 v1 재현 (통제 없음)

새 코드가 v1 수치를 그대로 내는지 먼저 확인한 기준선이다. 파라미터는 두 arm 모두 351,553.

| arm | recall@1 | **recall@5** | recall@10 | 검색 MRR | **completion acc@1** | acc@3 | wall/run | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| **hyper** | 0.511±0.044 | **0.863±0.022** | 0.933±0.015 | 0.667±0.030 | **0.502±0.015** | 0.756±0.015 | 103 s | 82 MB |
| reified_binary | 0.270±0.047 | 0.670±0.054 | 0.838±0.028 | 0.452±0.049 | 0.424±0.028 | 0.703±0.037 | 111 s | 101 MB |

v1 canonical run(recall@5 0.863 / 0.670, acc@1 0.502 / 0.424)과 소수 셋째 자리까지 같다. → **재현 확인**, verdict `success` (5/5, 5/5).

### v2a_inductive — 학습 그래프에서 held-out 사실의 구조를 제거

학습 스텝의 인코더 입력을 train split 사실만 담은 하이퍼그래프로 제한했다. 엔티티는 78개 전부 남기고(엔티티는 모두 알려진 것으로 둔다),
held-out 사실 118개와 그에 딸린 멤버십 498개만 뺐다. 즉 **평가할 사실의 구조를 모델이 학습 중에 본 적이 없다.**

| arm | **recall@5** | **completion acc@1** | acc@3 | 검색 MRR |
|---|---|---|---|---|
| **hyper** | **0.870±0.035** | **0.310±0.020** | 0.497±0.024 | 0.658±0.019 |
| reified_binary | 0.723±0.016 | 0.212±0.014 | 0.431±0.021 | 0.503±0.025 |

seed별 판정 쌍(`hyper` vs `reified_binary`):

| seed | recall@5 | acc@1 |
|---|---|---|
| 1 | 0.872 vs 0.752 | 0.297 vs 0.229 |
| 2 | 0.906 vs 0.726 | 0.288 vs 0.203 |
| 3 | 0.880 vs 0.709 | 0.305 vs 0.203 |
| 4 | 0.889 vs 0.718 | 0.314 vs 0.229 |
| 5 | 0.803 vs 0.709 | 0.347 vs 0.195 |

**verdict `success`** (5/5, 5/5). 부수 기록: transductive(v1) 대비 completion acc@1이 `hyper` 0.502 → 0.310,
`reified_binary` 0.424 → 0.212로 **둘 다 크게 떨어졌다.** 이는 예상된 것이고 실패가 아니다. 구조를 본 적 없는 사실을 보완하는 일이
훨씬 어렵기 때문이며, 중요한 것은 **떨어진 뒤에도 격차(+0.098)가 v1(+0.078)보다 오히려 커졌다**는 점이다. 검색은 질의 문장이
엔티티 이름을 담고 있어 구조 제거의 영향을 거의 받지 않았다(0.863 → 0.870).

### v2b_head — reified arm의 출력층을 hyper와 동일하게 맞춤

`reified_binary`가 blank node의 `node_out` 대신 `edge_out`을 통해 사실 벡터를 꺼내도록 고쳤다. `hyper` arm은 건드리지 않았으므로
`hyper` 수치는 v1_baseline과 **완전히 동일**해야 하고, 실제로 동일하다(요인이 하나만 바뀌었다는 확인).

| arm | **recall@5** | **completion acc@1** | acc@3 | v1_baseline 대비 |
|---|---|---|---|---|
| **hyper** | **0.863±0.022** | **0.502±0.015** | 0.756±0.015 | 변화 없음(설계상 동일) |
| reified_binary | 0.699±0.042 | 0.403±0.027 | 0.719±0.034 | 검색 0.670 → 0.699, 보완 0.424 → 0.403 |

seed별 판정 쌍 — recall@5: 0.821/0.718, 0.880/0.684, 0.872/0.632, 0.863/0.701, 0.880/0.761 (5/5).
acc@1: 0.500/0.407, 0.508/0.398, 0.475/0.356, 0.517/0.432, 0.508/0.424 (5/5). **verdict `success`.**

해석: 헤드를 맞추자 대조군의 검색은 조금 올랐지만(+0.029) 보완은 오히려 조금 내렸고(−0.021), 격차는 그대로 남았다.
**v1 격차의 원인이 출력층 비대칭이었다는 가설은 지지되지 않는다.**

### v2c_text — 동결 multilingual sentence-transformer 타워

해싱 n-gram 텍스트 인코더를 `paraphrase-multilingual-MiniLM-L12-v2`(동결, 학습되는 것은 선형 projection뿐)로 바꿨다.
기반 임베딩은 텍스트당 1회 계산해 캐시하므로 학습 예산이 동결 타워 쪽으로 새지 않는다. 동결 타워는 학습 대상이 아니므로
학습 파라미터 수는 97,345(`hyper`·`reified_binary`)·97,332(`flat_text`)로 줄어든다.

| arm | recall@1 | **recall@5** | recall@10 | 검색 MRR | **completion acc@1** | acc@3 |
|---|---|---|---|---|---|---|
| **hyper** (제안) | 0.368±0.037 | **0.846±0.020** | 0.942±0.016 | 0.575±0.029 | **0.405±0.024** | 0.656±0.022 |
| reified_binary (판정 대조) | 0.200±0.077 | 0.585±0.133 | 0.779±0.094 | 0.376±0.089 | 0.251±0.017 | 0.527±0.029 |
| flat_text (참조선, 판정 외) | 0.509±0.021 | 0.897±0.031 | 0.966±0.012 | 0.674±0.020 | 0.236±0.023 | 0.541±0.025 |

seed별 판정 쌍 — recall@5: 0.838/0.444, 0.812/0.701, 0.872/0.427, 0.855/0.761, 0.855/0.590 (5/5).
acc@1: 0.424/0.246, 0.390/0.254, 0.398/0.229, 0.441/0.280, 0.373/0.246 (5/5). **verdict `success`.**

**참조선이 말해주는 것**: 구조를 전혀 안 쓰는 `flat_text`가 검색에서 `hyper`보다 높다(0.897 vs 0.846). 이는 charter가 **결과를 보기 전에
예측해 적어둔 부작용**이며, 실패가 아니라 "검색은 구조 없이도 된다"는 v1 해석의 강화다. 같은 `flat_text`가 N항 보완에서는 0.236으로
`hyper`(0.405)에 크게 뒤진다. 즉 **강한 텍스트 타워는 검색을 텍스트 문제로 만들지만, 보완 문제를 대신 풀어주지는 못한다.**
`reified_binary`의 검색 분산이 유독 크다(±0.133, seed 1에서 0.444). 강한 타워 아래에서 reified 그래프의 학습이 seed에 민감했다는
관찰이며, 판정은 두 지표 모두 5/5이므로 이 변동에 의존하지 않는다.

### v2d_real — 실데이터 WD50K_100 (Wikidata qualifier 문장)

| 항목 | 값 |
|---|---|
| 문장(사실) 수 | 31,314 (train 22,738 / valid 3,279 / test 5,297) |
| 엔티티 / 관계 | 18,025 / 204 |
| 평균 항수 / 최대 항수 | 3.30 / 7 (**전 문장이 3항 이상**, n-ary 비율 1.0) |
| 프로토콜 | inductive(held-out 사실 8,576개, 멤버십 28,643개를 학습 그래프에서 제거) + 헤드 정합 |
| 평가 | split당 2,000개 표본(seed 0 고정, 두 arm 동일 표본), completion forward 하나가 가리는 사실 128개(=0.4%) |

이 데이터에는 자연어 문장이 없으므로 텍스트-구조 정렬 과제가 존재하지 않는다. 그래서 **retrieval 값은 의도적으로 0으로 기록**되며,
판정 지표는 charter가 미리 고정한 link prediction 표준 지표(MRR, Hits@10)다.

| arm | **MRR** | **Hits@10** | Hits@1 | Hits@3 | wall/run | peak VRAM |
|---|---|---|---|---|---|---|
| **hyper** | **0.147±0.005** | **0.289±0.008** | 0.078±0.005 | 0.151±0.006 | 503 s | 569 MB |
| reified_binary | 0.118±0.001 | 0.215±0.005 | 0.067±0.001 | 0.119±0.003 | 547 s | 970 MB |

seed별 판정 쌍(MRR, Hits@10): 0.144/0.119·0.293/0.213, 0.140/0.118·0.282/0.219, 0.155/0.119·0.294/0.220,
0.145/0.116·0.278/0.208, 0.152/0.119·0.300/0.219 — 두 지표 모두 **5/5**. **verdict `success`.**

절대값은 이 벤치마크의 문헌 수치보다 훨씬 낮다. 예산이 합성 월드와 같은 1,500 step·d=64이고 표본 채점을 쓰기 때문이며,
charter의 non-goal대로 **SOTA를 주장하지 않는다.** 여기서 읽을 수 있는 것은 "같은 인코더·같은 예산·같은 표본에서 구조 표현 두 가지의
상대 비교"뿐이다. 부수 관찰로 `reified_binary`가 VRAM을 1.7배 쓰고(970 vs 569 MB) 더 느렸다(547 vs 503 s):
같은 사실을 blank node와 2항 엣지로 풀면 그래프 자체가 커지기 때문이다.

### v2_combined — A·B·C 동시 적용 (보고용, 추가 판정 아님)

inductive + 헤드 정합 + 강한 타워를 한 번에 켠 조건. charter §8에 따라 **단독 판정 결과를 뒤집지 않으며 보고용**이다.

| arm | **recall@5** | recall@10 | **completion acc@1** | acc@3 |
|---|---|---|---|---|
| **hyper** | **0.815±0.067** | 0.938±0.024 | **0.234±0.032** | 0.473±0.033 |
| reified_binary | 0.605±0.029 | 0.791±0.021 | 0.178±0.024 | 0.444±0.030 |

seed별 — recall@5: 0.803/0.632, 0.829/0.632, 0.906/0.556, 0.701/0.615, 0.838/0.590 (5/5).
acc@1: 0.246/0.220, 0.229/0.161, 0.212/0.153, 0.195/0.186, 0.288/0.169 (5/5). verdict `success`.
통제를 모두 걸면 절대 보완 성능은 0.502(v1) → 0.234로 절반 아래가 되지만 **순서는 유지된다.**

### v2e_depth_parity — 깊이 × 헤드 정합 steelman (2026-09-11 추가)

규칙·예산·해석은 `docs/charter-hnn-ontology-encoder-v2e-depth-parity.md`에 **관측 전** 동결됐다.
`v2b_head`와 **완전히 같은 조건**에서 대조군만 층 2배 arm으로 바꾼 것이다.

동기: "격차는 표현이 아니라 **유효 깊이** 탓"이라는 반박이 절반만 닫혀 있었다. `hyper`는 한 사실의
참여자들이 한 층 안에서 만나고, `reified_binary`는 blank node를 거쳐 두 층이 걸린다. v1은 깊이를
**헤드 정합 없이** 쟀고(`reified_binary_deep`, 5/5 패, 격차 9%만 축소), v2b는 정합을 **깊이 없이** 쟀다.
둘을 동시에 건 조건은 없었다.

| arm | 파라미터 | **recall@5** | **completion acc@1** |
|---|---|---|---|
| **hyper** (L=2) | 351,553 | **0.863±0.025** | **0.502±0.016** |
| reified_binary_deep + 정합 (L=4) | 401,729 (**+14%**) | 0.672±0.066 | 0.400±0.039 |

seed별 acc@1 — `hyper` / 깊은 대조군 / (참고) v2b의 얕은 대조군:
0.500/0.441/0.407, 0.508/0.432/0.398, 0.475/0.407/0.356, 0.517/0.364/0.432, 0.508/0.356/0.424.
**verdict `success`** (recall@5 5/5, acc@1 5/5).

**격차 축소율 −3.4%.** charter §4가 관측 전에 정한 기준 격차는 `v2b_head`의 +0.0983이고, 깊은 대조군에서
격차는 **+0.1017**로 오히려 조금 **벌어졌다**. 사전 등록된 읽기의 첫 칸(<25%)에 해당한다:
**깊이와 헤드 정합을 동시에 줘도 격차는 설명되지 않는다.**

부수 관찰 둘. 깊이는 **균일하게 나쁘지 않다** — seed 1·2·3에서는 얕은 대조군을 이기고(0.441/0.432/0.407 vs
0.407/0.398/0.356) seed 4·5에서 크게 진다(0.364/0.356 vs 0.432/0.424). 표준편차가 0.030 → 0.039로 커졌다.
층을 늘린 것이 **평균을 올리는 대신 분산을 키웠다**는 관측이며, 이 실행이 그 원인을 규명하지는 않는다.
그리고 깊은 대조군은 정합을 켜자 v1의 0.431에서 0.400으로 내려갔다 — 정합이 얕은 대조군을 낮춘 것과 같은 방향이다.

## 해석과 주장 경계

1. **주장하는 것**: 같은 정보·같은 인코더·같은 파라미터·같은 초기 가중치·같은 예산에서, N항 사실을 하이퍼엣지 하나로 **보존**하는 표현이
   같은 사실을 blank node + 2항 엣지로 **푸는** 표준 reification보다 N항 보완에서 일관되게 낫다. 이 우위는 (a) 구조를 학습에서 감춰도,
   (b) 대조군의 출력층을 같게 맞춰도, (c) 텍스트 타워를 강하게 바꿔도, (d) 합성이 아닌 Wikidata 실데이터에서도 유지됐다(각각 5/5 seed).
2. **주장하지 않는 것 — 검색**: `flat_text` 참조선이 검색에서 `hyper`보다 높다(0.897 vs 0.846). **"구조가 검색을 개선한다"는 주장은
   하지 않는다.** 검색 recall@5는 두 구조 표현을 비교하는 사전등록 지표로만 쓰고, 절대 우수성의 근거로 쓰지 않는다.
3. **주장하지 않는 것 — 절대 성능**: v2d의 MRR 0.147은 이 벤치마크에서 경쟁력 있는 수치가 아니다. 예산과 표본 채점 때문이며, SOTA 주장은
   charter의 non-goal이다. 절대 성능이 아니라 **동일 조건 상대 비교**가 이 실험의 산출물이다.
4. **여전히 열린 것**: 온톨로지는 두 종류(합성 1, Wikidata 1)만 봤고 인코더 규모는 하나(d=64, 2층)뿐이다. 예산을 키웠을 때 격차가
   유지·확대·축소되는지는 이 실험이 답하지 않는다. 템플릿 밖 비정형 텍스트에서의 추출도 다루지 않았다.

## 표현별 강점 분해 (판정 밖 진단, 3 seed)

`.harness/scripts/probe_representation_split.py`가 세 표현을 **같은 월드·같은 예산(1,500 step)·같은 QA test 118문항**에서
비교한다. 판정이 아니라 "무엇이 무엇을 잘하는가"를 보기 위한 참고 측정이며, 동점을 argmax로 처리하므로
판정 지표(동점 비관 처리)보다 관대하다.

| 표현 | 검색 hit@5 | 보완 acc@1 (전체 후보 중) | 종합 (top-5 안에서 답 고르기) |
|---|---|---|---|
| `hyper` (N항을 하이퍼엣지로 보존) | 0.486 | **0.746** | **0.528** |
| `reified_binary` (blank node + 2항 엣지) | 0.280 | 0.734 | 0.421 |
| `flat_text` (문장 임베딩, 일반 벡터 인덱스와 같은 것) | **0.890** | 0.345 | 0.370 |

읽는 법은 두 가지다.

1. **일반 임베딩과는 강점의 방향이 반대다.** `flat_text`는 검색에서 `hyper`를 거의 두 배로 앞서고(0.890 vs 0.486),
   보완에서는 절반 이하로 뒤진다(0.345 vs 0.746). 사실을 **찾는** 일과 참여자를 **엮는** 일은 다른 능력이며,
   본문의 v2c 관측(`flat_text` 검색 0.897 vs `hyper` 0.846)과 같은 방향이다.
2. **표준 reification과의 차이는 이 진단에서 검색·종합에 몰려 있다.** 보완 acc@1만 보면 0.746 vs 0.734로
   seed 3개 안에서 `reified_binary`의 분산이 커(0.644 / 0.822 / 0.737) 차이가 노이즈에 묻힌다. 판정(§v2b, 5/5)은
   동점을 비관적으로 처리하고 split 전체 completion 질의를 쓰는 **더 엄격한 측정**이며, 이 진단은 그것을 대체하지 않는다.

가설로 남는 것: 검색은 텍스트가, 답 고르기는 하이퍼그래프가 잘하므로 둘을 결합하면 양쪽을 다 이길 여지가 있다.
**이 문서는 그것을 측정하지 않았다.** 하려면 charter를 먼저 쓰고 조건을 동결해야 한다.

## 운영 기록

- `v2c_text`의 `flat_text` 참조선을 이미 판정이 끝난 같은 출력 디렉터리에 이어 돌렸을 때, 집계 함수가 그 프로세스가 만든 행만으로
  `summary.json`·`judgment.json`을 덮어썼다. seed별 summary는 남아 있어 복구했고, 집계를 **디스크의 seed summary에서 다시 읽도록**
  고쳐 나중에 arm을 추가하는 것이 교체가 아니라 확장이 되게 했다(커밋 `4479cc38f`). 판정 규칙·예산·seed는 변경되지 않았다.
  이 사건은 `correct(+25)`로 이벤트 로그에 기록됐다.
- 판정 수치는 전부 커밋된 산출물에서 나온다(커밋 `3f3d68d5b`). 어떤 seed도 verdict를 본 뒤 다시 돌리지 않았다.

## 재현

```
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v1_baseline   --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2a_inductive --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2b_head      --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2c_text      --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2d_real      --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2_combined   --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
```

기본값이 charter §7-b의 동결 예산과 같다(`--steps 1500 --batch-size 32 --lr 2e-3 --d 64 --layers 2 --text-dim 64 --text-buckets 4096
--real-eval-limit 2000 --mask-chunk 128`, deterministic 기본). 산출물은
`outputs/hnn_ontology_encoder_v2/{factor}/{arm}/seed_{k}/{summary.json,history.csv}`와 factor별
`summary.{json,csv}`·`judgment.json`·`run_config.json`, 실행 로그 `outputs/hnn_ontology_encoder_v2_run.log`.

## 다음 단계 (2026-09-09 갱신)

**아래의 원래 다음 단계는 실행됐고 닫혔다. 이 절만 갱신했으며 위의 판정·수치·verdict는 불변이다.**

원안: *"Paper 2(LLM bridge) P5 실행: `docs/charter-hnn-llm-bridge-v1.md`의 동결 판정 규칙으로 5 arm을 비교한다."*

실제로 일어난 일: Paper 2는 v1부터 v6까지 여섯 사이클을 돌았고 **모두 실패했다.** v6 charter가 관측 전에 정한
정지 규칙에 따라 **0토큰 구조 전달 주장은 철회**됐고 v7은 만들지 않는다(`docs/hnn-llm-bridge-v6-results.md` §8).
제안 arm은 검색된 사실의 이름만 나열한 참조선(0.400)을 한 번도 넘지 못했다.

**이 문서의 Paper 1 판정은 그 실패와 독립적이다** — 위 여섯 factor는 인코더 자체의 채점이며 LLM을 쓰지 않는다.
다만 "그러므로 LLM에 도움이 된다"로 확장해 읽으면 안 되며, 그 경계는
`docs/hnn-ontology-paper1-consolidation.md` §4에 적었다.

**지금의 다음 단계**: Paper 1 정리. 진행 상황은 `docs/hnn-ontology-paper1-consolidation.md`,
범위는 `docs/charter-hnn-ontology-paper1-consolidation.md`.
