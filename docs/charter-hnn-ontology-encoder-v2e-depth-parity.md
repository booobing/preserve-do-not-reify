# Charter — encoder v2e: 깊이 × 헤드 정합 steelman (사전등록)

작성 2026-09-10. **결과는 아직 관측되지 않았다.** 판정 규칙·arm·seed·해석 규칙을 여기서 동결한다.
선행: `docs/hnn-ontology-encoder-v1-results.md`(깊이 steelman), `docs/hnn-ontology-encoder-v2-results.md`(헤드 정합),
`docs/charter-hnn-ontology-encoder-v2.md` §7(판정 규칙).

Ambiguity **LOW** — 질문이 하나다. Difficulty **Simple** — 기존 arm·기존 러너, factor 하나 추가, 약 20분.

## 1. 왜 이것만 남았나

`hyper`가 이기는 이유가 **표현**이 아니라 **유효 깊이**라는 반박이 가능하다. `hyper`는 한 사실의 참여자들이
한 층 안에서 만나지만, `reified_binary`는 참여자 → blank node → 다른 참여자로 **두 층**이 걸린다.

이 반박은 **절반만 닫혀 있다.**

| 조건 | 깊이 2배 | 헤드 정합 | 실행됨? | 결과 |
|---|---|---|---|---|
| v1 steelman `reified_binary_deep` | ○ | ✕ | ✅ v1 | 보완 격차 +0.078 → **+0.071**(9%만 좁힘), `hyper` 5/5 승 |
| `v2b_head` | ✕ | ○ | ✅ v2 | reified 보완 0.424 → **0.403**(오히려 하락) |
| **깊이 + 정합 동시** | ○ | ○ | **❌ 없음** | — |

각각은 격차를 못 메웠지만 **둘을 합친 조건은 측정된 적이 없다.** 그것만 채운다.

## 2. 무엇을 돌리나

**신규 factor `v2e_depth_parity`**: `inductive=False, head_parity=True, text_tower="hashing", real=False`
— 즉 `v2b_head`와 **완전히 같은 조건**에서 대조군만 깊은 arm으로 바꾼다.

- 제안: **`hyper`** (L=2, 파라미터 351,553)
- 대조: **`reified_binary_deep`** (L=4, 파라미터 401,729, **헤드 정합 켬**)

**파라미터 정합은 의도적으로 깨진다.** 대조군이 층 2개와 파라미터 14%를 더 갖는다.
원 판정의 공정성 조건(같은 파라미터)은 여기서 **대조군에게 유리한 쪽으로** 완화되며, 그것이 steelman의 목적이다.

## 3. 판정 규칙 (동결)

`docs/charter-hnn-ontology-encoder-v2.md` §7과 **같은 규칙**을 그대로 쓴다:
`hyper`가 **`retrieval.recall@5`와 `completion.acc@1` 두 지표 모두에서 ≥4/5 seed** 우위여야 `success`.
동점은 우위로 세지 않는다. seed `1 2 3 4 5`, 예산은 §7-b 동결값(1,500 step, batch 32, lr 2e-3, d=64, deterministic).

**이것은 논문의 새 판정이 아니라 기존 주장에 대한 강화 통제다.** 다만 빈 게이트가 아니다:
**`hyper`가 여기서 ≥4/5를 못 얻으면 논문의 주장은 한정돼야 한다**(§5의 해석 규칙).

## 4. 해석 규칙 — 관측 전에 고정

기준 격차는 `v2b_head`의 보완 acc@1 격차 **+0.099**(hyper 0.502 vs reified 0.403)다.
새 격차를 `g`라 할 때 **격차 축소율 = (0.099 − g) / 0.099**로 읽는다.

| 축소율 | 사전 등록된 읽기 |
|---|---|
| **< 25%** | 깊이 + 정합은 설명이 **아니다.** 표현 차이라는 해석이 유지되고, 이 결과를 논문의 방어선으로 싣는다. |
| **25 ~ 75%** | **부분 설명.** 축소율을 그대로 보고하고, 논문은 "표현 차이의 일부는 유효 깊이로 설명된다"로 **약화해서** 쓴다. |
| **> 75%** | 깊이가 **사실상의 설명**이다. 논문 프레이밍을 다시 써야 한다. |

`hyper`가 4/5를 못 얻는 경우는 축소율과 무관하게 **주장 한정** 사유다.

## 5. 사전 등록 예측 (틀릴 수 있게 적는다)

- `hyper`가 두 지표 모두 **5/5로 이길 것**으로 본다.
- 축소율은 **25% 미만**일 것으로 본다. 근거: 깊이 단독이 9%만 좁혔고(v1), 정합 단독은 대조군을 **낮췄다**(−0.021).
- 이 예측이 맞아도 그것은 "표현이 이유다"의 **증명이 아니라 대안 설명 하나의 배제**다.

## 6. Non-goals

- v1·v2 verdict 수정 없음. 이 factor는 **추가**이며 기존 판정을 대체하지 않는다.
- 합성 월드 `v1`만. inductive·실데이터와의 결합은 이번 범위가 아니다(필요하면 별도 charter).
- 새 지표 없음. 새 arm 없음(`reified_binary_deep`는 v1부터 있던 arm이다).
- Paper 2 재개 아님 — 이것은 Paper 1 강화이며 v6 정지 규칙과 무관하다.

## 7. Touched files

- 수정 `src/ontology/run_encoder_benchmark_v2.py` — `FACTORS`에 `v2e_depth_parity` 추가,
  `judge()`가 factor별 대조 arm을 받도록(기본값은 종전 `JUDGED_ARMS`).
- 수정 `tests/test_ontology_v2.py` — 새 factor의 대조 arm이 `reified_binary_deep`이고
  다른 factor의 판정 쌍은 바뀌지 않는지 고정.
- 산출물 `outputs/hnn_ontology_encoder_v2/v2e_depth_parity/`
- 결과는 `docs/hnn-ontology-encoder-v2-results.md`에 절 추가 + 초안·정리 문서 반영.

## 8. Required checkpoints과 예산

`charter_frozen` → `v2e_contract_tests` → `v2e_run`(5 seed × 2 arm) → `results_doc` → `checkpoint_gate`.

**예산**: 실측 기준 `hyper` 103 s/run, 깊은 arm은 층이 2배이므로 130~150 s/run 예상 →
5 seed × 2 arm ≈ **20분**. GPU 하드 리셋이 나도 `--resume`으로 잇는다(이 러너는 seed 단위로 기록한다).
