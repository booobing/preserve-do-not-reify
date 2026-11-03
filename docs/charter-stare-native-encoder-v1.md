# Charter — 5장 인코더를 StarE 프로토콜로: 보존 / reification / 한정쌍 제거 v1 (사전등록)

작성 2026-10-21. **어떤 결과도 아직 관측되지 않았다.** 사용자 지시 "작성하고 실험해"(2026-10-21)로 작성·실행하며, 이 작업에 한해 Clarification Debt 정지(132.5)를 해제한 것으로 기록한다.

선행: `docs/related-work-hkg-comparison.md` §6의 1번 제안, `docs/charter-stare-input-representation-v1.md`(StarE 하네스·판정 규칙), `docs/stare-input-representation-v1-results.md`(StarE 백본 판정 `hyper_wins`).

Ambiguity **LOW** — 하네스·판정 규칙·예산이 이미 있고 인코더만 바꾼다. Difficulty **Complex** — 장시간 GPU(15 실행), 새 인코더 경로.

## 1. 왜 이 실험인가

선행 연구 비교에서 세 가지가 남았다.

1. **우리 5장 인코더의 절대 수치를 문헌 표에 놓을 수 없다.** 5장의 WD50K(100) MRR 0.147은 가리는 위치·후보·예산·채점이 표준 프로토콜과 모두 달랐다.
2. **StarE 백본 판정에서 배제 못 한 설명**: StarE 원본은 한정쌍을 엣지 특징으로 쓰고 한정값이 메시지를 받지 않는다. 우리 하이퍼엣지는 모든 참여자를 대칭으로 다룬다. 둘의 차이와 "보존 대 reification"의 차이가 섞여 있었다.
3. **"Understanding"(arXiv 2508.03280)의 반론**: 한정쌍을 아예 버린 NBFNet이 WD50K에서 HKG 전용 모델과 비슷했다. 우리 실험에는 한정쌍 제거 대조군이 없었다.

이 실험은 **5장 인코더의 메시지 전달 층을 그대로** StarE 파이프라인(데이터·sampler·Transformer 디코더·filtered 평가)에 넣고, 인코더가 보는 그래프만 세 가지로 바꾼다.

## 2. 고정 사실 (실행 전에 센 값)

WD50K(100), 판정 split(StarE 기본: 그래프 = train + valid 26,017문장, 평가 = test 5,297 × 2방향 = 10,594 질의).

| arm | 인코더 그래프 | 노드 | 하이퍼엣지 | 멤버십 | 파라미터 |
|---|---|---|---|---|---|
| `native_hyper` | 문장마다 하이퍼엣지 1개(주술어 타입), 멤버 = 주어·목적어·한정값 전부(역할 = subject / object / 한정 속성) | 18,792 | 26,017 | 89,208 | 5,318,216 |
| `native_reified` | 5장 `reify()`: 문장 노드 + 멤버마다 2항 역할 엣지 | 44,809 (문장 노드 26,017) | 89,208 | 178,416 | 5,318,416 |
| `native_prune` | 문장마다 2항 하이퍼엣지(주어·목적어), 한정쌍은 인코더에 없음 | 18,792 | 26,017 | 52,034 | 5,318,216 |

- 세 arm은 **같은 모듈**이다: 엔티티 표(StarE `init_embed`), 디코더 관계 표(StarE `init_rel`), 엣지 타입 표 560, 역할 표 281, `RoleAwareMessageLayer` 2층(d = 200), StarE Transformer 디코더. reified만 문장 노드 공통 오프셋 200개가 더 있다.
- **디코더 입력 `[s, r, q₁, v₁, …]`은 세 arm이 같다.** `native_prune`에서도 디코더는 한정쌍 토큰을 본다. 이 실험이 바꾸는 것은 인코더 그래프뿐이다.
- 비교 기준(같은 프로토콜, 이미 관측): StarE 원본 0.6553 ± 0.0013, StarE 위 reified 그래프 0.6485 ± 0.0021(`docs/stare-input-representation-v1-results.md`). 우리 arm은 StarE 원본보다 파라미터가 약 4.9% 많다(250,200개, 대부분 메시지 층).
- 재부팅 뒤 엔티티는 18,792(패딩 포함), 후보 10,874, 한정값 전용 제외 7,918(패딩 포함). 모두 StarE 하네스와 같다.

## 3. 예산과 운영

- StarE 기본값 그대로: **401 epoch**, batch 128, Adam lr 1e-4, StepLR(50, 0.95), 라벨 스무딩 0.1, grad clip 1.0, dropout은 StarE 위치(1층 뒤 0.3, 2층 뒤 0.3 `HID_DROP`/`FEAT_DROP`).
- 평가: 10 epoch마다와 401에서 test. **판정은 401 epoch 값.** test 최고값은 참고로만.
- seed `1 2 3 4 5`, 순서 seed 우선(seed 1: hyper → reified → prune, seed 2: …). 리셋이 와도 seed가 완결된 짝부터 남는다.
- 운영: step당 150 ms GPU 휴지, 체크포인트 5 epoch, 드라이버 `run_stare_judged.ps1`(완료 실행 건너뜀, 부분 실행 재개). 사용자 클럭 상한(`nvidia-smi -lgc 300,1400`)·전력 한도(`-pl 160`)를 권장한다.
- 파일럿은 두지 않는다. 예산은 StarE 기본값으로 이미 고정이고, 파일럿 v1이 보인 것처럼 짧은 예산은 순서를 잘못 보여 준다. 스모크(arm마다 2 epoch + 재개 1회)로 시간만 재고 §3-b에 적는다.

### 3-b. 스모크 실측 (2026-10-21, 판정 실행 전)

`outputs/stare_native_encoder_v1/smoke/`, seed 0, 판정 split, arm마다 2 epoch(reified는 `--resume`으로 3 epoch까지 이어서 재개 확인). 스모크의 test 지표는 읽지 않았다.

| arm | epoch당 벽시계(휴지 포함) | 학습 연산(휴지 제외) | test 평가 1회 | peak VRAM | 실행당 예상 |
|---|---|---|---|---|---|
| `native_hyper` | 83.3 s | 32.1 s | 2.5 s | 1,309 MB | 401 × 83.3 + 41 × 2.5 ≈ **9.3 h** |
| `native_reified` | 89.6 s | 38.4 s | 4.1 s | 2,850 MB | ≈ **10.0 h** |
| `native_prune` | 83.0 s | 31.9 s | 2.2 s | 945 MB | ≈ **9.3 h** |

- 재개: reified가 epoch 2 체크포인트에서 `resumed_from_epoch = 2`로 이어 3 epoch를 마쳤다.
- 15 실행 합계 ≈ 143시간(약 6일). 시작 2026-10-21 저녁 기준 종료 예상 2026-10-27 저녁. Windows Update 일시 중지는 2026-11-25까지다.
- 스모크 시점에 GPU는 재부팅 뒤 기본값(최대 2,100 MHz, 200 W)이었다. 사용자에게 클럭 상한·전력 한도를 다시 요청했다.
- 예산·규칙은 §3·§4 그대로다. 이 표는 시간 기록이며 아무것도 바꾸지 않는다.

## 4. 판정 규칙 (동결, 양측)

규칙은 StarE 판정과 같다: test **MRR과 Hits@10**, 401 epoch. seed마다 엄격히 큰 arm이 이긴다. 두 지표 모두 5 seed 중 4 이상을 이긴 쪽의 이름이 verdict이고, 아니면 `no_consistent_order`. 방향은 미리 정하지 않는다.

| 판정 | 비교 | 질문 |
|---|---|---|
| **J1 (주)** | `native_hyper` vs `native_reified` | 같은 인코더·같은 디코더·같은 예산에서 보존이 reification을 이기는가 — 5장 주장을 표준 프로토콜에서 다시 묻는다 |
| **J2** | `native_hyper` vs `native_prune` | 한정쌍을 인코더 구조로 넣는 것이 버리는 것보다 나은가 — "Understanding"의 반론 |

보고(판정 아님): seed별 표, 쌍 차이의 평균·95% 구간·t p·정확 부호 뒤집기 p, Hits@1·Hits@3·MR, 목적어/주어 예측 분리, 곡선, 그리고 StarE 원본(0.6553)·StarE 위 reified(0.6485)·문헌 표(`related-work-hkg-comparison.md` §2-a)와의 **위치**.
위치 비교는 seed·코드가 다른 실행 사이의 비교라 판정이 아니다.

판정 명령:

```
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other native_reified
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other native_prune
```

## 5. 사전 예측 (틀릴 수 있게)

- **J1**: `native_hyper_wins` 60%, `no_consistent_order` 30%, `native_reified_wins` 10%. 근거: 5장 인코더에서 7조건 5/5였고 StarE 백본에서도 같은 방향이었다. 다만 StarE 백본 격차가 0.007로 작았다.
- **J2**: `no_consistent_order` 45%, `native_hyper_wins` 40%, `native_prune_wins` 15%. 근거: WD50K(100)은 모든 문장에 한정쌍이 있어 한정쌍의 효과를 보기에 가장 유리한 자리지만(StarE H 0.654 대 T 0.380), 디코더가 이미 한정쌍 토큰을 보므로 인코더 구조의 추가 몫은 작을 수 있다.
- **위치**: `native_hyper`는 StarE 원본(0.6553)보다 **낮을** 것으로 본다(55%). GLoRE(KDD'26)가 노드 단위 집계의 역할 혼동을 보고했고, 우리 층은 엔티티당 표현이 하나인 평균 집계다. 0.60~0.67 범위로 예측한다.

## 6. 읽기 규칙 (관측 전 고정)

| 결과 | 읽기 |
|---|---|
| J1 `native_hyper_wins` | 5장의 주장이 표준 프로토콜·표준 예산에서도 성립한다. 논문 5장에 "세 번째 확인(표준 프로토콜)"으로 싣는다 |
| J1 `no_consistent_order` 또는 `native_reified_wins` | 5장의 7조건 결과는 **5장 프로토콜·예산에 한정**된다. 논문 주장을 그 범위로 줄여 쓴다 |
| J2 `native_hyper_wins` | 한정쌍을 인코더 구조로 넣는 것이 디코더 토큰만으로는 대신되지 않는다 |
| J2 그 밖 | "Understanding"의 반론이 이 설정에서도 맞다. 한정쌍 구조의 몫은 디코더가 이미 흡수한다고 적는다 |
| 위치 | 어느 값이든 그대로 표에 넣는다. StarE보다 낮으면 "통제 비교의 인코더이지 경쟁 모델이 아니다"를 명시한다 |

arm·예산·규칙은 결과를 본 뒤 바꾸지 않는다. 결과가 이상해 보여도 버그 사냥으로 재실행하지 않고, 재실행이 필요하면 새 charter를 쓴다.

## 6-b. 결과 (2026-10-27 20:14, 15 실행 완료, 규칙·예산·arm 불변)

- **J1 `native_hyper_wins`**: 보존이 reification을 MRR·Hits@10 모두 5/5로 이겼다. 쌍 차이 MRR −0.0087 [−0.0110, −0.0065], Hits@10 −0.0092 [−0.0106, −0.0077].
- **J2 `native_hyper_wins`**: 보존이 한정쌍 제거를 MRR·Hits@10 모두 5/5로 이겼다. 쌍 차이 MRR −0.0129 [−0.0171, −0.0087], Hits@10 −0.0069 [−0.0107, −0.0031].
- 위치: `native_hyper` test MRR 0.6648 ± 0.0026 — StarE 원본 0.6553보다 높아 §5의 위치 예측(낮을 것 55%)은 틀렸다. §6 읽기 규칙대로 5장 주장은 표준 프로토콜에서도 성립한다고 적는다.
- 운영: 10-22 하드 리셋 1회(Windows Update 재시작으로 GPU 상한 해제) → epoch 30에서 재개, 이후 리셋 0회. 결과 문서 `docs/stare-native-encoder-v1-results.md`.

## 7. Non-goals

- 하이퍼파라미터 탐색 없음. d·층 수·dropout은 StarE 기본값, 메시지 층은 5장 그대로.
- 5장 V2-D·StarE 판정의 수치·verdict 수정 없음.
- WD50K 전체·JF17K·WikiPeople·WD50K(33/66) 없음 — 후속.
- HAHE·KREPE 등 외부 모델 실행 없음 — 후속(`related-work-hkg-comparison.md` §6의 2번).
- 디코더 질의 쪽 reification 없음(StarE 판정 §4-4와 같은 이유).

## 8. Touched files

- 신규 `src/ontology/stare_bridge/native_encoder.py`, `tests/test_stare_native.py`(9), 이 charter, 결과 `docs/stare-native-encoder-v1-results.md`.
- 수정 `src/ontology/stare_bridge/runner.py`(arm 3개 추가), `.harness/scripts/judge_stare_input_representation.py`(임의의 arm 쌍; 기본 쌍의 출력은 불변, 커밋된 StarE 판정을 재현하는 테스트로 확인), `.gitignore`(새 출력 폴더의 체크포인트·로그).
- 갱신 `docs/charter.md`, `docs/checkpoints.md`, 이벤트 로그.
- 산출물 `outputs/stare_native_encoder_v1/{smoke,judged}/{arm}/seed_{k}/`.

## 9. Required checkpoints

`charter_frozen` → `native_contract`(테스트 + Ruff) → `smoke_three_arms`(GPU, 재개 1회) → `budget_addendum`(§3-b) → `judged_run`(15 실행) → `judged_verdict`(J1, J2) → `results_doc` → `checkpoint_gate` → `clarification_debt`.

## 10. Rollback

신규 파일 삭제, `runner.py`의 arm 추가와 판정 스크립트 일반화 되돌림. 기존 StarE 판정 산출물은 읽기만 하므로 영향이 없다.
