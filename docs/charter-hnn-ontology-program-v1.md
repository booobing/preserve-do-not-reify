# Charter — HNN 기반 온톨로지 프로그램 v1 (프로그램 charter)

작성: 2026-08-20. 이 charter는 이후 두 편의 논문 작업을 묶는 **프로그램 수준 charter**다.
개별 실험은 이 파일의 하위 charter(§A, §B)를 사전등록 규칙으로 삼고, 실행 전 동결한다.

## 1. Objective

하이퍼그래프 + HNN을 **온톨로지(클래스·N항 관계·공리)의 네이티브 표현**으로 쓰는 시스템을 두 단계로 만든다.

- **Paper 1 (Encoder)**: 정형(레코드/표)·비정형(텍스트) 데이터를 하이퍼그래프로 바꾸고, 역할(role)을 아는
  HNN으로 노드·하이퍼엣지 임베딩을 만든 뒤, CLIP처럼 텍스트 타워와 대조학습으로 정렬한다.
  판정 질문: *같은 정보를 이진(reified) 그래프로 풀었을 때보다, 하이퍼엣지로 보존했을 때 검색·N항 보완
  성능이 좋은가?*
- **Paper 2 (LLM bridge)**: 동결된 사전학습 Transformer에 HNN 임베딩을 gated cross-attention으로 주입해,
  텍스트 직렬화 없이 검색·제약 검사·추론을 forward 안에서 수행한다.
  판정 질문: *같은 LLM·같은 검색 예산에서, 텍스트 직렬화 RAG(vector/GraphRAG/HyperGraphRAG-text)보다
  N항 QA 정확도·제약 위반률·토큰 비용이 좋은가?*

## 2. Non-goals

- Tetris 실험 코드(`src/models`, `src/training`, `src/eval`)의 동작을 바꾸지 않는다. 재사용은 import만 한다.
- EGGROLL/미분 없는 최적화를 새 프로그램의 주장 축으로 쓰지 않는다. Encoder는 backprop(Adam)으로 학습한다.
- 이번 사이클에서 실제 LLM 가중치를 불러오는 실행(HF `transformers`)을 **required gate**로 두지 않는다.
  Paper 2는 이번 사이클에서 계약·어댑터 모듈·워크플로우까지만 만들고, LLM 결합 실험은 P4 이후 별도 charter로 연다.
- 외부 네트워크 의존(모델 다운로드, 외부 KG 덤프)을 기본 경로에 두지 않는다. 기본 경로는 오프라인 합성 온톨로지다.
- W3C 표준(RDF/OWL/SPARQL) 호환을 주장하지 않는다. 온톨로지 스키마는 이 저장소의 JSON 계약(`docs/hnn-ontology-contracts-v1.md`)만 따른다.

## 3. Touched files

- 신규 패키지 `src/ontology/**` (schema, hypergraph, synthetic, extract, text_encoder, encoder, contrastive, baselines, evaluate, run_encoder_benchmark, llm_bridge/)와 `src/ontology/AGENTS.override.md`
- 신규 테스트 `tests/test_ontology_*.py`
- 신규 문서 `docs/charter-hnn-ontology-program-v1.md`, `docs/hnn-ontology-design-v1.md`, `docs/hnn-ontology-contracts-v1.md`, `docs/hnn-ontology-workflow-v1.md`, `docs/charter-hnn-llm-bridge-v1.md`
- 갱신 `docs/charter.md`(상단 addendum), `docs/checkpoints.md`, `.harness/checkpoints/default.yml`, `README.md`(프로그램 섹션), `progress.md`, `.harness/memory/session-events.jsonl`
- 신규 하네스 스크립트 `.harness/scripts/run_ontology_gates.py`
- 생성물은 `.harness/tmp/ontology_*`(smoke)과 `outputs/hnn_ontology_encoder_v1/*`(acceptance run)만 쓴다.

## 4. Ambiguity: MEDIUM → 가정 명시 후 보수적 옵션으로 진행

- "CLIP 같은 Encoder"를 *하이퍼그래프 타워(HNN) ↔ 텍스트 타워 대조 정렬*로 해석한다. 대안(텍스트→하이퍼그래프 생성
  모델 seq2seq)은 v2 과제로 남긴다.
- 비정형 텍스트 추출기는 v1에서 **결정적 템플릿/슬롯 추출기**다. LLM 기반 추출기는 프로토콜만 두고 구현하지 않는다.
- 데이터는 **합성 온톨로지 월드**(스키마·공리·레코드·문장·질의가 함께 생성되고 정답이 있는 세계)를 기본으로 한다.
  실데이터 ingest는 `StructuredRecordExtractor`의 매핑 파일로 열어 두되 acceptance에는 넣지 않는다.

## 5. Difficulty: Complex

새 도메인 패키지 + 새 하네스 + 사전등록 비교 + 테스트. planner 역할은 이 charter/design 문서가 맡고,
구현 뒤 `reviewer`·`qa`·`security-reviewer` 서브에이전트를 호출한다.

## 6. Required checkpoints (이번 사이클)

`charter_updated`, `ontology_contracts_frozen`, `ontology_lint`, `ontology_unit`(schema/hypergraph invariants),
`ontology_extract_fidelity`, `ontology_encoder_smoke`(CLI가 summary를 쓴다), `ontology_benchmark_acceptance`
(seed 5개 × 3 arm, judging rule 적용), `llm_bridge_adapter_contract`(gate=0이면 출력 불변), `docs_update`,
`checkpoint_gate`, `clarification_debt`.

## 7. Rollback plan

`src/ontology/`, `tests/test_ontology_*.py`, 신규 docs, `.harness/scripts/run_ontology_gates.py`를 제거하고
`docs/charter.md`·`docs/checkpoints.md`·`.harness/checkpoints/default.yml`·`README.md`의 addendum/섹션을 되돌린다.
Tetris 경로는 건드리지 않으므로 기존 gate는 영향이 없다.

---

## §A. Paper 1 사전등록 — Encoder (HNN 타워 vs 이진 reified 타워 vs flat text)

### A-1. 고정 요인
동일 **사실 집합**, 동일 **텍스트 타워**(hashing n-gram encoder, 동일 seed), 동일 **임베딩 차원 d**, 동일
**학습 step 수·배치·학습률·온도**, 동일 **seed 집합**. 파라미터 수는 arm 간 ±5% 이내로 맞추고 summary에 기록한다.

#### A-1-b. 동결된 acceptance 예산 (2026-08-20, dev seed 0 파일럿 뒤·판정 seed 관측 전)
`--world-preset v1`(세계 seed 1 고정, 사실 390개, train/val/test = 272/59/59), `--steps 1500 --batch-size 32 --lr 2e-3`,
`--d 64 --layers 2 --text-dim 64 --text-buckets 4096`, `--seeds 1 2 3 4 5`, device RTX 3060 Ti(cuda).
근거: dev seed `0`(판정 seed 집합 밖) 400 step 파일럿에서 `hyper` 정렬 loss가 아직 내려가는 중(0.32)이었고
`reified_binary`는 2.1 부근에서 느리게 내려갔다. 두 arm 모두 더 수렴하도록 약 4배(1500)로 고정한다. 파일럿 결과는
`.harness/tmp/ontology_pilot_v1b/`에 남기되 판정에 쓰지 않는다. 보조(advisory) arm `reified_binary_deep`(layer 4, 파라미터 더 많음)은
steelman으로 함께 돌리지만 §A-4 판정에는 들어가지 않는다.
절차 보강(2026-08-20, 1차 run 뒤·2차 run 관측 전): CUDA 비결정성이 확인되어 CLI 기본을 deterministic 모드로 바꾸고 같은 예산으로 재실행한다. 1차 run은 `outputs/hnn_ontology_encoder_v1_nondet_run1/`에 보존하며 두 run의 verdict를 모두 보고한다. 판정 규칙·예산·seed는 바꾸지 않는다.

세계 설계 보강(파일럿 전 확정): 사실 역할이 서로 독립이면 completion이 우연 수준에 갇히므로, 합성 월드에 **잠재 규칙성**을 넣었다
(설비→홈 구역·부착 센서, 기술자→담당 구역, 센서 종류→조치, 오류 심각도→조치; 각 사실은 확률 0.9로 규칙을 따른다). 규칙성은
노드 `attrs`에만 있고 노드 텍스트에는 없으므로 **다른 사실을 통해서만** 학습할 수 있다(`docs/hnn-ontology-design-v1.md` §7).

### A-2. 변동 요인 (1요인)
| arm | 구조 입력 | 설명 |
|---|---|---|
| `hyper` | N항 하이퍼엣지 + role | 제안. 사실 하나 = 하이퍼엣지 하나 |
| `reified_binary` | 같은 사실을 blank node + 2항 role 엣지로 분해 | 같은 정보·같은 인코더. 차이는 집계 방식뿐 |
| `flat_text` | 구조 없음 | 사실을 문장으로 직렬화해 텍스트 타워끼리 대조 |

`hyper`와 `reified_binary`는 **정보량이 같다**(이 저장소의 hyperedge ablation 교훈을 그대로 따른다).
차이가 나면 그것은 메커니즘(하이퍼엣지 집계 vs 쌍별 전파) 차이다.

### A-3. 지표 (우선순위 순)
1. `retrieval_recall@5`, `mrr`: 자연어 질의 → 정답 하이퍼엣지 (held-out 질의)
2. `completion_acc@1`: 하이퍼엣지 참여자 하나를 가리고 후보 노드 중 정답 맞히기 (held-out 사실)
3. `extraction_f1`: 추출기 단계의 하이퍼엣지 정확 복원(정형 1.0 기대, 텍스트 템플릿 ≥0.95)
4. 파라미터 수, step당 wall-clock(RTX 3060 Ti), peak VRAM

### A-4. 판정 규칙 (결과 관측 전 동결)
- seeds `1 2 3 4 5`, 각 arm 동일 seed.
- **성공**: `hyper`가 `reified_binary`를 `retrieval_recall@5`와 `completion_acc@1` **둘 다** 5 seed 중 ≥4에서 이김.
- **부분 성공**: 둘 중 하나만 ≥4/5. 문서에 "부분"으로 적는다.
- **실패**: 위 미만. 판정 규칙을 바꾸지 않고 실패로 적는다. "통과할 때까지 재실행"은 하지 않는다.
- `flat_text`는 참조선이다. `hyper`가 `flat_text`보다 못하면 그 자체로 보고한다(구조가 도움 안 됨).

### A-5. 산출물
`outputs/hnn_ontology_encoder_v1/{arm}/seed_{k}/summary.json`, 루트 `encoder_benchmark_summary.json/csv`,
`judgment.json`(규칙 적용 결과). smoke는 `.harness/tmp/ontology_encoder_smoke/`.

### A-6. 결과 (2026-08-20, 관측 후 추가; 규칙 변경 없음)
verdict **`success`**(결정 모드 canonical run): `hyper` > `reified_binary` recall@5 5/5, completion acc@1 5/5. 평균 recall@5 0.863 vs 0.670, acc@1 0.502 vs 0.424. 비결정 모드 1차 run도 `success` 5/5·5/5(0.863 vs 0.694, 0.502 vs 0.431).
`flat_text` 참조선은 recall@5 0.862(동률), acc@1 0.336. 상세·주장 경계: `docs/hnn-ontology-encoder-v1-results.md`.

## §B. Paper 2 사전등록 — LLM bridge (요약; 전체는 `docs/charter-hnn-llm-bridge-v1.md`)

> **상태: 닫힘 (2026-09-09).** v1~v6 여섯 사이클 전부 실패했고, v6 charter가 관측 전에 정한 정지 규칙에 따라
> **0토큰 구조 전달 주장은 철회**됐다. v7은 만들지 않는다. 결과: `docs/hnn-llm-bridge-v6-results.md` §8.
> 아래 사전등록 내용은 **기록으로 남기며 수정하지 않는다.**

arm: `llm_only`, `vector_rag`, `graph_rag_triples`, `hypergraph_rag_text`, `hnn_xattn`(제안).
지표: N항 QA EM/F1, 다중제약 QA 정확도, 공리 위반률, KB 밖 답변률(환각), 프롬프트 토큰, TTFT.
이번 사이클 범위: 어댑터 모듈 계약 + 합성 QA 생성 + 워크플로우. LLM 실행은 P4에서 별도 charter로 동결 후 시작.
