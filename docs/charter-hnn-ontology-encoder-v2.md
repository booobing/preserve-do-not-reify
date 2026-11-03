# Charter — HNN 온톨로지 Encoder v2 (Paper 1 확장, 사전등록)

작성 2026-08-20. **어떤 v2 결과도 아직 관측되지 않았다.** v1 결과는 `docs/hnn-ontology-encoder-v1-results.md`,
프로그램 charter는 `docs/charter-hnn-ontology-program-v1.md`.

## 1. 배경 — v1이 남긴 네 개의 구멍

v1은 합성 온톨로지 월드에서 `hyper`가 `reified_binary`를 두 지표 5/5 seed로 이겼다(verdict `success`).
그 결과를 리뷰하면서 네 가지 한계가 문서로 남았고, v2는 그 넷을 각각 하나의 요인으로 닫는다.

| 항목 | v1 상태 | v2에서 바꾸는 것 |
|---|---|---|
| A. 프로토콜 | transductive. test 사실의 **구조**가 학습 그래프에 있었다 | inductive. 학습 그래프에서 held-out 사실의 하이퍼엣지를 제거 |
| B. 헤드 정합 | `hyper`는 `edge_out`, `reified_binary`는 `node_out`을 사실 표현에 사용 | 두 arm 모두 `edge_out` 계열로 통일 |
| C. 텍스트 타워 | 오프라인 hashing n-gram(약함) | 동결 sentence-transformer + 학습 projection |
| D. 데이터 | 합성 월드 1개 | 실데이터 WD50K_100(Wikidata qualifier 문장) 추가 |

## 2. Objective

v1의 결론(“N항 사실을 하이퍼엣지로 보존하면 같은 정보를 이진 reification으로 푼 것보다 낫다”)이
**네 개의 통제를 각각 통과하는지** 1회씩 판정한다. 통과하지 못하면 통과하지 못한 것으로 적는다.

## 3. Non-goals

- 판정 규칙·예산·seed를 결과를 본 뒤에 바꾸지 않는다. 바꿔야 하면 v3 charter를 새로 만든다.
- Tetris 경로(`src/models`, `src/training`, `src/eval`)를 건드리지 않는다.
- Paper 2(LLM bridge) 실행은 이 charter의 범위가 아니다(`docs/charter-hnn-llm-bridge-v1.md`).
- WD50K에서 SOTA를 주장하지 않는다. 같은 인코더·같은 예산 아래 **구조 표현 두 가지의 상대 비교**만 한다.
- 텍스트 타워를 파인튜닝하지 않는다. 동결 + 선형 projection만 학습한다.

## 4. Touched files

- 신규 `src/ontology/inductive.py`(학습용 부분 그래프 구성), `src/ontology/real_data.py`(WD50K 로더),
  `src/ontology/run_encoder_benchmark_v2.py` 또는 기존 CLI의 v2 플래그
- 수정 `src/ontology/baselines.py`(헤드 정합), `src/ontology/text_encoder.py`(ST 임베딩 캐시),
  `src/ontology/contrastive.py`(inductive view)
- 신규 테스트 `tests/test_ontology_inductive.py`, `tests/test_ontology_real_data.py`
- 신규 문서 이 charter, 결과는 `docs/hnn-ontology-encoder-v2-results.md`
- 갱신 `docs/hnn-ontology-contracts-v1.md`(계약 추가분), `docs/checkpoints.md`, `.harness/checkpoints/default.yml`
- 생성물 `outputs/hnn_ontology_encoder_v2/{factor}/{arm}/seed_{k}/`, 원시 데이터 `data/wd50k_100/`

## 5. Ambiguity: LOW / Difficulty: Complex

v1의 계약·하네스·판정 절차를 그대로 재사용하고 요인만 바꾸므로 설계 불확실성은 낮다. 다만 실데이터 로더,
inductive 분리, ST 캐시가 각각 새 코드이고 규모가 v1의 60배(사실 390 → 22,738)라 난이도는 Complex다.

## 6. 공통 고정 요인 (모든 v2 실험)

`hyper`와 `reified_binary`는 **같은 인코더 모듈·같은 Vocab·같은 파라미터 수·같은 seed의 동일 초기 가중치·
같은 학습 예산·같은 텍스트 타워**를 쓴다. v1에서 테스트로 고정한 공정성 불변식(가린 멤버 은닉, 학습 텍스트
split 분리, 동일 초기 가중치)은 v2에서도 그대로 유지하며, 새 코드가 이를 깨지 않는지 테스트로 확인한다.
deterministic 모드가 기본이고 seed는 `1 2 3 4 5`로 v1과 같게 두어 v1 수치와 직접 대조할 수 있게 한다.

## 7. 요인별 사전등록

### V2-A. Inductive 프로토콜

- 바꾸는 것: 학습 스텝의 인코더 입력을 **train split 사실만 포함한 하이퍼그래프**로 제한한다. 평가 시에만
  전체 그래프를 넣는다. 노드 집합은 전체를 유지하므로(엔티티는 모두 알려져 있음) 달라지는 것은 held-out
  사실의 하이퍼엣지 존재 여부뿐이다.
- 판정 지표: v1과 동일하게 test split의 `retrieval_recall@5`, `completion_acc@1`.
- **성공**: `hyper`가 `reified_binary`를 두 지표 모두 ≥4/5 seed에서 이김.
- **부분/실패**: v1 §A-4와 같은 정의.
- 부수 기록: v1(transductive) 대비 두 arm의 절대 성능 하락폭. 하락 자체는 실패가 아니다.

### V2-B. Reified 헤드 정합

- 바꾸는 것: `ReifiedArm.encode_facts`가 blank node의 `node_out` 대신 `edge_out(node_hidden[blank])`을
  쓰고, `completion_query`도 같은 hidden을 넣는다. 이로써 두 arm의 readout 경로 구조가 동일해진다.
- 이것은 **대조군을 강화하는 수리**다. 격차가 줄어드는 방향이 기대되며, 줄어들어도 정직하게 기록한다.
- 판정 지표: 위와 동일.
- **성공**: 헤드 정합 후에도 `hyper`가 두 지표 모두 ≥4/5 seed 우위.
- **실패 시 해석**: v1 격차의 일부는 메커니즘이 아니라 readout 비대칭이었다는 뜻이며, 그렇게 적는다.

### V2-C. 강한 텍스트 타워

- 바꾸는 것: `HashingTextEncoder` → `SentenceTransformerTextEncoder(paraphrase-multilingual-MiniLM-L12-v2)`,
  동결 + 학습 가능한 선형 projection. 모든 arm이 같은 타워 인스턴스 구조를 쓴다. 기반 임베딩은 텍스트당 1회
  계산해 캐시한다(학습 예산 왜곡 방지).
- 모델 선정 근거: 합성 월드 문장이 한국어·영어 혼합이므로 다국어 모델이 필요하다. 캐시된 `all-MiniLM-L6-v2`는
  영어 전용, `ko-sroberta-multitask`는 한국어 전용이라 둘 다 한쪽 언어를 불리하게 만든다.
  `paraphrase-multilingual-MiniLM-L12-v2`는 apache-2.0, gated 아님, 두 언어를 모두 처리한다.
- 판정 지표: 위와 동일.
- **성공**: 강한 타워에서도 `hyper`가 두 지표 모두 ≥4/5 seed 우위.
- **예상되는 부작용을 미리 적는다**: v1에서 `flat_text`는 이미 검색에서 `hyper`와 동률이었다. 강한 타워는
  `flat_text`의 검색을 더 올릴 가능성이 높다. 그것은 실패가 아니라 “검색은 구조 없이도 된다”는 v1 해석의
  강화이며, 판정은 `hyper` vs `reified_binary`에서만 한다. `flat_text`는 참조선으로 계속 보고한다.

### V2-D. 실데이터 (WD50K_100)

- 데이터: `github.com/migalkin/StarE` `data/clean/wd50k_100/statements/{train,valid,test}.txt`
  (train 22,738 / valid 3,279 / test 5,297, **전 문장이 qualifier를 가진 N항**). 형식은
  `주어,주술어,목적어[,한정속성,한정값]*`이며, 이를 하이퍼엣지 하나로 매핑한다:
  멤버 = `(주어, role=subject)`, `(목적어, role=object)`, 각 한정쌍마다 `(한정값, role=한정속성)`.
  하이퍼엣지 타입 = 주술어. reified view는 기존 `reify()`를 그대로 쓴다(Wikidata의 statement 노드와 같은 구조).
- 텍스트가 없는 데이터이므로 **정렬(CLIP) 없이 구조 과제만** 수행한다. 텍스트 타워를 쓰지 않으므로
  V2-C와 독립이다.
- 과제: N항 completion(표준 link prediction). test 문장마다 멤버 하나를 가리고 전체 엔티티 중 순위를 매긴다.
  학습에 존재하는 다른 정답은 제외하는 filtered ranking을 쓰고, 동점은 v1과 같이 비관적으로 처리한다.
- 판정 지표: `MRR`, `Hits@10` (이 벤치마크의 표준 지표).
- **성공**: `hyper`가 `reified_binary`를 `MRR`과 `Hits@10` 모두 ≥4/5 seed에서 이김.
- **부분**: 둘 중 하나만. **실패**: 그 미만.
- 예산은 별도 파일럿(dev seed `0`, 판정 seed 밖)에서 정하고, 판정 실행 전에 이 charter에 추가 기록한다.

## 7-b. 예산 동결 (2026-08-21, dev seed `0` 파일럿 뒤·판정 seed 관측 전)

합성 월드 요인(A/B/C/결합)은 v1과 같은 예산을 쓴다: `--world-preset v1 --steps 1500 --batch-size 32 --lr 2e-3
--d 64 --layers 2 --text-dim 64 --text-buckets 4096`, seeds `1 2 3 4 5`, deterministic 기본, RTX 3060 Ti.
근거는 dev seed `0` 파일럿이다. `v2a_inductive`는 hyper `0.863/0.347` vs reified `0.744/0.254`,
`v2c_text`는 hyper `0.769/0.407` vs reified `0.607/0.288`로 두 arm 모두 loss가 1 아래로 수렴했고
실행당 91~111초였다. 파일럿 산출물은 `.harness/tmp/ontology_v2_pilot/`에 두고 판정에는 쓰지 않는다.

V2-D는 같은 step/batch를 쓰되 텍스트가 없으므로 completion loss만 학습한다(`train_structural`).
평가는 held-out 문장에서 split당 최대 2,000개를 seed 0으로 고정 추출해 채점하고, completion forward 하나에
가리는 사실 수를 128개로 제한한다. 전수 채점 대신 표본을 쓰는 이유는 비용이며, 두 arm이 **동일한 표본**을 본다.
한 forward에서 여러 사실을 동시에 가리면 후보를 채점하는 그래프가 그만큼 흔들리므로, 128/31,314 = 0.4%로 묶어
왜곡을 제한한다. 이 값들은 판정 전에 고정하며 결과를 본 뒤 바꾸지 않는다.

## 8. 요인 결합

A~C를 각각 단독으로 판정한 뒤, 셋을 모두 켠 결합 조건을 1회 더 돌린다. 결합 조건은 **추가 판정이 아니라
보고용**이며, 단독 판정 결과를 뒤집지 않는다. 결합에서만 결과가 뒤집히면 그 사실을 한계로 적는다.

## 9. Required checkpoints

`charter_frozen`, `inductive_contract`, `head_parity_contract`, `text_tower_cache_contract`,
`real_data_loader_contract`, `v2_unit_tests`, `v2_pilot_budget`(dev seed 0), `v2a_run`, `v2b_run`, `v2c_run`,
`v2d_run`, `v2_combined_run`, `results_doc`, `checkpoint_gate`, `clarification_debt`.

## 10. Rollback plan

신규 파일(`inductive.py`, `real_data.py`, v2 CLI, v2 테스트, 이 charter, 결과 문서)을 제거하고
`baselines.py`·`text_encoder.py`·`contrastive.py`의 수정을 되돌린다. v1 산출물과 판정은 영향받지 않는다.
