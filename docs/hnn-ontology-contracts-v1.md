# HNN 온톨로지 — 계약 v1 (동결: 2026-08-20)

테스트(`tests/test_ontology_*.py`)가 검증하는 대상이다. 계약을 바꾸면 이 문서·테스트·`docs/checkpoints.md`를 한 패치에서 함께 바꾼다.

## C1. OntologySchema (JSON)

```json
{
  "name": "industrial_maintenance_v1",
  "entity_types": ["Equipment", "Sensor", "Zone", "Technician", "ErrorCode", "Action"],
  "relation_types": [
    {"name": "DefectEvent",
     "roles": [{"name": "equipment", "type": "Equipment"},
               {"name": "sensor_primary", "type": "Sensor"},
               {"name": "sensor_secondary", "type": "Sensor"},
               {"name": "zone", "type": "Zone"},
               {"name": "action", "type": "Action"}]}
  ],
  "axioms": [
    {"name": "z3_vibration_emergency", "relation": "DefectEvent",
     "if": {"zone": {"name_eq": "Z3"}, "sensor_secondary": {"attr_eq": ["kind", "vibration"]}},
     "then": {"action": {"name_eq": "EmergencyStop"}}}
  ]
}
```
불변식: entity_types 중복 없음, relation 이름 중복 없음, 각 role 타입은 entity_types에 속함, arity = len(roles) ≥ 2,
axiom의 relation/role 이름은 실제로 존재.

## C2. Hypergraph (메모리 객체 + JSON)

- `Node(id: str, type: str, name: str, text: str = "", attrs: dict = {})`
- `Hyperedge(id: str, relation: str, members: list[tuple[node_id, role]], text: str = "", provenance: str = "")`
- 불변식(`Hypergraph.validate(schema)`): 멤버 노드 존재, relation 존재, role 집합 == schema role 집합(순서 무관, 중복 없음),
  각 멤버 노드 타입 == role 타입(단 `__blank__`/`__role_edge__` 예약 타입은 reified 구조에서만 허용), 빈 하이퍼엣지 금지.
- `to_incidence(vocab) -> IncidenceBatch`(`vocab = schema.vocab()`): `node_type_ids[V]`, `edge_type_ids[E]`, `incidence[nnz,2]`(node_idx, edge_idx; 정렬),
  `role_ids[nnz]`, `node_index: dict[id,int]`, `edge_index: dict[id,int]`. nnz == Σ arity.
- `reify() -> Hypergraph`: 하이퍼엣지 e(arity n) → blank node `__b__{e.id}`(type `__blank__`) + n개 2항 엣지
  (relation `__role_edge__{role}`, members `[(blank,"subject"),(member,"object")]`). 노드 수 V+E, 엣지 수 Σ arity.
  reified 그래프는 원래 role 이름을 relation 이름으로 운반하므로 **정보 손실 0**이다.
- JSON round-trip: `Hypergraph.to_json()` ↔ `Hypergraph.from_json()` 동치.

## C3. 합성 월드 (`synthetic.generate_world(config) -> World`)

- 입력: `WorldConfig(seed, n_equipment, n_sensors, n_zones, n_technicians, n_error_codes, n_facts={relation: count}, languages=("en","ko"), max_sentence_templates, retrieval_queries_per_fact, completion_roles_per_fact, split, n_axiom_probes)`; 프리셋 `WORLD_PRESETS["smoke"|"v1"]`.
- 출력 `World`: `schema`, `hypergraph`(정답), `records: list[dict]`(`relation` + role 컬럼), `sentences: list[Sentence]`
  (`edge_id, lang, template_id, text`), `queries: list[Query]`(`id, edge_id, kind ∈ {retrieval, completion}, text, lang, masked_role?, answer_node_id?`),
  `axiom_probes: list[AxiomProbe]`(`id, axiom, edge(dict), text`: 공리를 위반하는 가짜 사실), `splits`(`train/val/test` → edge_id 리스트, 사실 단위·관계별 분할), `entity_catalog`.
- 결정성: 같은 config → 같은 World(JSON 동치). 테스트가 보장.
- 잠재 규칙성(`REGULARITY_P=0.9`): 설비 `home_zone`/`sensors`, 기술자 `zones`(이 세 속성은 `attrs`에만 있고 노드 `text`에 들어가지 않는다), 그리고 센서 `kind`→action, 오류 `severity`→action(이 두 속성은 노드 text에 드러나며 공리 평가에도 쓰인다).

## C4. 추출기

- `StructuredRecordExtractor(schema, mapping).extract(records) -> Hypergraph`. mapping: `{relation: {role: column}}`(기본은 동일 이름).
  같은 (type,name)은 같은 노드 id로 합친다. 정형 레코드에서 **정확 복원 F1 == 1.0**이 계약.
- `TemplateTextExtractor(schema, templates).extract(sentences) -> Hypergraph`. 슬롯 문법으로 역매칭. 계약: 합성 월드 문장에서 F1 ≥ 0.95.
- `LLMExtractor`는 `Protocol`만 정의(`extract(texts) -> Hypergraph`), 구현 없음.
- `extraction_fidelity(pred, gold) -> {precision, recall, f1}`: 하이퍼엣지 키 = (relation, frozenset((node_type,node_name,role))).

## C5. 인코더 I/O

- `RoleAwareHypergraphEncoder(num_node_types, num_edge_types, num_roles, d, layers, text_dim)`
- `forward(batch: IncidenceBatch, node_text_emb[V,text_dim]) -> {"node_emb":[V,d], "edge_emb":[E,d], "node_hidden":[V,d], "edge_hidden":[E,d]}` (emb는 L2 정규화, `finite`)
- `completion_query(hidden[M,d], role_ids[M]) -> [M,d]`(L2 정규화): `completion_out([hidden ; E_role[r]])`. `hyper`는 edge_hidden, `reified_binary`는 blank node_hidden을 넣는다.
- V=0 또는 E=0도 오류 없이 빈 텐서를 돌려준다. CPU/CUDA 동일 코드.
- `count_parameters(model)`를 summary에 기록. 판정 쌍 `hyper`/`reified_binary`는 파라미터 수가 **동일**해야 하고(같은 모듈·같은 Vocab), `flat_text`는 인코더 파라미터 수에 맞춘 MLP 어댑터로 ±5% 이내.

## C6. 텍스트 타워

- `TextEncoder(texts: list[str]) -> Tensor[n, text_dim]`(`nn.Module.forward`); `HashingTextEncoder(num_buckets, text_dim, ngram=(1,2), char_ngram=(3,))`.
  같은 입력·같은 seed → 같은 출력. 빈 문자열도 허용(0 벡터 아님, bias만).
- `SentenceTransformerTextEncoder`는 import 실패 시 명확한 `ImportError` 메시지(설치 방법)로 실패한다. 테스트는 skip.

## C7. 정렬/학습 (`contrastive.AlignmentModel`)

- `loss = 0.5·(CE(s·sim, arange) + CE(s·simᵀ, arange))`, `s = exp(logit_scale)`(= 1/τ) 학습, 초기 `s = 1/0.07`, 클램프 `[1, 100]`.
- 총 loss = InfoNCE(사실↔텍스트) + `completion_weight`(기본 1.0)·completion CE(가린 멤버 vs 같은 타입 후보). 세 arm 동일.
- `train_alignment(model, view, world, TrainConfig(steps, batch_size, lr, weight_decay, completion_weight, seed, log_every), device=) -> history(list[dict(step, loss, loss_align, loss_completion, temperature, elapsed_s)])`. 마지막 loss < 첫 loss(smoke 계약).
- 학습 텍스트는 **train split 사실의 문장 + 검색 질의만** 쓴다. val/test 사실의 텍스트는 학습에 들어가지 않는다.
- 프로토콜은 transductive다: 그래프(모든 사실의 구조)는 학습 중 전체가 보이고, held-out은 (a) 텍스트↔사실 짝과 (b) completion에서 가린 멤버다. 평가 시 completion은 forward당 사실마다 최대 1개 멤버를 가리며(동일 forward에 여러 test 사실이 동시에 가려짐), 모든 arm에 동일하게 적용한다.

## C8. 평가 (`evaluate.py`)

- `retrieval_metrics(query_emb[Q,d], edge_emb[E,d], gold_idx[Q], ks=(1,5,10)) -> {recall@k..., mrr}`
- `completion_metrics(...) -> {acc@1, acc@3}`: 가린 멤버 후보 = 같은 타입의 모든 노드.
- 순위는 비관적(pessimistic)이다: 정답과 동점인 후보는 정답보다 앞선 것으로 센다(붕괴된 임베딩이 1.0을 받을 수 없다).
- 모든 값은 `[0,1]` float. Q=0이면 `nan`이 아니라 `0.0`과 `n=0`을 기록.

## C9. CLI 계약 (`python -m src.ontology.run_encoder_benchmark`)

인자: `--arms hyper reified_binary flat_text [reified_binary_deep] --seeds 1 2 3 4 5 --steps N --out DIR --device auto|cpu|cuda --world-preset smoke|v1 [--no-deterministic]`. 기본은 deterministic 모드(`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`)이며 같은 seed·같은 GPU에서 bit 동일 결과를 낸다. `--out`은 `.harness/tmp/` 또는 `outputs/` 아래만 허용(`--allow-any-out`로 해제).
산출: `DIR/{arm}/seed_{k}/summary.json`(`arm, seed, params, steps, final_loss, retrieval{...}, completion{...}, extraction{...}, wall_time_s, peak_vram_mb`),
`DIR/encoder_benchmark_summary.json`(전 행), `DIR/encoder_benchmark_summary.csv`, `DIR/judgment.json`(§A-4 규칙 결과:
`wins_recall5, wins_completion, verdict ∈ {success, partial, fail, incomplete}`). smoke: `--world-preset smoke --steps 30` 이 30초 내 완료.

## C10. LLM bridge 계약 (`llm_bridge/`)

- `HypergraphMemory.build(node_emb, edge_emb) -> (memory[1,E+V,d_llm], mask[1,E+V])`
- `GatedCrossAttention(d_model, d_memory, n_heads)`: `forward(x[B,T,d_model], memory, mask) -> [B,T,d_model]`;
  초기 `alpha=0` → `forward(x,...) == x` (allclose atol 0).
- `inject_into_decoder(toy_decoder, layer_ids, ...)`: 주입 후 `alpha=0`에서 logits 동일. 주입 대상 레이어 외 파라미터 `requires_grad=False`.

- `llm_bridge/rag_baselines.py`: `build_context(graph, edge_ids, style ∈ {vector_rag, graph_rag_triples, hypergraph_rag_text})`, `build_prompt(question, context, lang)`, `approx_token_count`(P5에서 실제 토크나이저로 교체).
- `llm_bridge/qa_data.py`: `build_qa_items(world)`(completion 질의 → `QAItem{question, answer, gold_edge_ids, masked_role, n_constraints, multi_constraint(≥3), lang, split}`), `build_axiom_probe_items(world)`(`forbidden_answer`/`required_answer`/`axiom`), `export_qa_dataset(world, dir)` → `qa_{train,val,test}.jsonl`, `axiom_probes.jsonl`, `qa_summary.json`.

## C13. v2 요인 (charter-hnn-ontology-encoder-v2)

- `schema.Role(name, type, optional=False)`. `RelationType.required_role_names` / `min_arity`. 하이퍼엣지의 role 집합은
  선언된 role의 **부분집합**이어야 하고 required role은 모두 있어야 한다. 합성 월드는 전부 required이므로 v1 불변식과 동일하게 동작한다.
- `inductive.build_training_views(view, train_fact_ids)`: 노드 전체 + train 사실의 하이퍼엣지만 담은 `train_native`/`train_reified`를
  붙인다. arm은 `self.training`일 때만 이 view를 읽는다. `coverage_report(view)`가 제거된 사실·멤버십 수를 보고한다.
- `baselines.ModelConfig.head_parity`: True면 `ReifiedArm`이 blank node hidden을 `edge_out`으로 읽어 `hyper`와 같은 헤드를 쓴다.
  파라미터 수는 두 값에서 동일하다.
- `text_encoder.CachedSentenceTransformerTower`: 동결 ST + 학습 projection. 문자열당 기반 임베딩을 1회만 계산해 캐시한다.
- `real_data.load_wd50k(variant)`: WD50K 문장 → 하이퍼엣지(`subject`/`object` required, qualifier property가 optional role).
  `data/<variant>/`에 1회 내려받고 이후 오프라인. `RealDataset.stats()`가 arity 분포를 보고한다.
- `evaluate.completion_metrics(..., max_masks_per_forward, with_mrr)`: forward당 동시에 가리는 사실 수를 제한하고 MRR을 추가한다.
- CLI `run_encoder_benchmark_v2 --factor {v1_baseline,v2a_inductive,v2b_head,v2c_text,v2_combined,v2d_real}`:
  `DIR/{factor}/{arm}/seed_{k}/summary.json`, `DIR/{factor}/{summary.json,summary.csv,judgment.json,run_config.json}`.
  판정 지표는 group-qualified(`retrieval.recall@5`, `completion.acc@1`; 실데이터는 `completion.mrr`, `completion.acc@10`).

## C12. HF bridge (`llm_bridge/hf_adapter.py`, P5)

- 고정 상수: `BRIDGE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"`, `BRIDGE_MODEL_REVISION="989aa7980e..."`,
  `TEXT_TOWER_MODEL_ID="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`, `DEFAULT_INJECTION_LAYERS=(7,14,21)`.
  charter가 단일 진실 소스이며 코드에서 임의로 바꾸지 않는다.
- `load_bridged_decoder(d_hnn, ...) -> BridgedDecoder`: 기본 `HF_HUB_OFFLINE=1`, fp16, 디코더 전체 `requires_grad=False`,
  지정 레이어에 gated cross-attention hook 부착. `decoder_layers(model)`은 transformers의 여러 레이아웃을 지원한다.
- 계약: `set_memory(...)` 후 gate 0이면 logits가 **bit 동일**, gate를 열면 달라진다. backward 시 grad는 어댑터·memory에만
  생기고 디코더 파라미터는 `grad is None`이다. fp32 인코더 출력이 fp16 memory/어댑터에 들어가도 캐스팅으로 처리된다.
- `load_text_tower(...)`: 동결 sentence-transformer. 모든 파라미터 `requires_grad=False`.
- 검증 아티팩트: `outputs/hnn_llm_bridge_v1/model_pin_verification.json`.

## C14. Paper 2 실행 (P6)

- `llm_bridge/retriever.py`: `FactRetriever.train(world, view, seed, steps)` → 학습된 Paper 1 `hyper` 타워를 동결해 재사용.
  `top_k(question, k)`는 결정적이며 **모든 arm이 같은 결과를 받는다**. `memory_inputs(fact_ids)`는 그 사실들의 엣지 임베딩과
  참여 노드 임베딩을 돌려준다(텍스트 arm이 직렬화하는 것과 같은 증거). `retrieval_hit_rate`는 arm 실패와 검색 실패를 분리하기 위한 보조 지표.
- `llm_bridge/generation.py`: `greedy_generate`가 toy·HF 두 디코더에 같은 루프를 쓴다. `ttft_ms`는 첫 토큰을 만드는 prefill forward의
  벽시계 시간이고, CUDA에서는 동기화 후 측정한다. `prompt_tokens`는 실제 토크나이저 기준이다.
- `llm_bridge/qa_eval.py`: `axiom_violation_rate`는 **공리가 요구하는 값 대신 다른 구체적 답을 말한 비율**이며, "모름"은 위반이 아니라
  `axiom_abstain_rate`로 따로 센다. `out_of_kb_rate`도 같은 이유로 abstention을 제외한다.
- CLI `run_bridge_benchmark`: arm 5종, `--decoder toy|hf`(toy는 같은 토크나이저를 쓰는 소형 랜덤 디코더로 GPU 없이 전 경로를 돌린다).
  산출물 `DIR/{arm}/seed_{k}/{summary,predictions}.json`, `DIR/bridge_benchmark_summary.{json,csv}`, `DIR/judgment.json`.
  판정은 `multi_constraint_acc`↑와 `axiom_violation_rate`↓ 둘 다 ≥4/5 seed **그리고** 평균 `prompt_tokens`가 더 적을 때만 `success`.

## C11. 하네스 gate

`docs/checkpoints.md`의 `ontology_*` gate와 `.harness/scripts/run_ontology_gates.py`(gate 일괄 실행, PASS/FAIL 표). 생성물은
`.harness/tmp/ontology_*`. acceptance run만 `outputs/hnn_ontology_encoder_v1/`.
