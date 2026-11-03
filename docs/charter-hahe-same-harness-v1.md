# Charter — HAHE를 같은 하네스로: 5장 인코더 대 HAHE v1 (사전등록)

작성 2026-10-28. **HAHE 쪽 결과는 아직 하나도 관측되지 않았다.** 사용자 지시 "HAHE도 같은 하네스로 돌리는 charter 쓰고 실험해"(2026-10-28)로 작성·실행하며, 이 작업에 한해 Clarification Debt 정지(132.5)를 해제한 것으로 기록한다.

선행: `docs/charter-stare-native-encoder-v1.md`(Study 3, 하네스·판정 규칙), `docs/stare-native-encoder-v1-results.md`(`native_hyper` test MRR 0.6648 ± 0.0026), `docs/related-work-hkg-comparison.md` §2-a·§6.

Ambiguity **MEDIUM** — 아래 세 가지는 선택이 필요했고, 각각 가장 보수적인 쪽을 골랐다.

| 선택 | 후보 | 고른 것과 이유 |
|---|---|---|
| 데이터 | (a) HAHE 저장소의 WD50K(100) json, (b) StarE 하네스가 우리 모델에 넣는 행 그대로 | **(b)**. "같은 하네스"의 뜻이 이것이다. (a)는 valid·test는 같지만 train이 1문장 적다(10-28 대조) |
| 평가 | (a) HAHE 자체 규칙(모든 엔티티가 후보, 동점은 불리하게), (b) StarE 규칙(주어·목적어로 등장한 엔티티만 후보, 이중 argsort) | **(b)로 판정, (a)는 보고만.** `native_hyper`가 (b)로 판정됐으므로 비교가 성립하는 쪽은 (b)뿐이다 |
| 학습 설정 | (a) HAHE README의 WD50K 명령, (b) 우리가 탐색한 설정 | **(a) 그대로.** WD50K(100)용 공개 설정은 없다. 탐색은 하지 않는다(§7). GLoRE 저자들은 재실행 때 탐색을 했으므로 이 차이를 한계로 적는다 |

Difficulty **Complex** — 외부 코드 고정·이식, 장시간 GPU(5 실행), 논문 표 수정.

## 1. 왜 이 실험인가

논문 Table 5와 `related-work-hkg-comparison.md` §2-a에서 우리 인코더(0.6648)의 위치는 **GLoRE 논문 Figure 5(a)를 읽은 값**에 기대고 있다. HAHE ≈ 0.670, MAYPL ≈ 0.658이고, 두 값 모두 원저자가 아니라 GLoRE 저자들이 다시 돌린 결과다. 이 값에는 세 가지 불확실성이 있다.

1. 그림을 읽은 값이라 ±0.003 오차가 있다.
2. GLoRE 쪽 데이터 전처리·후보 집합·필터 규칙이 우리 하네스와 같은지 알 수 없다.
3. 우리 인코더와 차이가 0.005 안팎이라, 프로토콜 차이만으로도 순서가 뒤집힐 수 있다.

같은 데이터 행·같은 질의·같은 후보·같은 채점식으로 HAHE를 직접 돌리면 2와 3이 사라진다. 그러면 "우리 인코더는 HAHE보다 위인가 아래인가"를 판정 규칙으로 물을 수 있다.

## 2. 고정 사실 (실행 전에 센 값, 2026-10-28)

HAHE 원본: `https://github.com/LHRLAB/HAHE.git` commit `e35e0a7b`(MIT), `third_party/HAHE`에 고정, `third_party/hahe.lock.json`이 `src/*.py` 다섯 파일의 sha256을 고정한다. 원본 코드는 한 줄도 고치지 않는다. `run.py`는 import 시점에 인자를 읽으므로 import하지 않고, 학습 루프를 `src/ontology/hahe_bridge/runner.py`가 그대로 옮긴다.

데이터: StarE 하네스(`stare_bridge.upstream.load_data`, WD50K(100), 한정쌍 최대 6쌍, 중복 유지)의 행을 그대로 쓴다.

| 항목 | 값 | `native_hyper`와 같은가 |
|---|---|---|
| 학습 그래프 | train 22,738 + valid 3,279 = 26,017문장 | 같음 |
| 평가 | test 5,297문장 × 주어·목적어 = 10,594 질의 | 같음 |
| 후보 엔티티 | 10,874 (한정값으로만 나오는 7,917 + 패딩 제외) | 같음 |
| 필터 | (주어 또는 목적어, 관계, 한정쌍 원래 순서 그대로) 키로 train+valid+test의 다른 정답 제거 | 같음 |
| 순위 | `1 + argsort(argsort(−score))[정답]`, 제외·필터 항목은 −1e6 | 같음 |
| 지표 | 목적어 예측과 주어 예측의 평균(StarE `_mean_metrics_`) | 같음 |

HAHE 쪽으로 옮긴 결과:

- 어휘 19,071 = `[PAD]`, `[MASK]` + 관계 278 + 엔티티 18,791. HAHE 저장소의 WD50K(100) 어휘 크기와 같다. 엔티티 토큰 = 2 + 278 + (StarE id − 1)로 1:1 대응하며, 테스트가 이 대응을 확인한다.
- 문장 arity(2 + 한정쌍 수) 분포(전체 31,314문장): 3: 20,426 / 4: 9,342 / 5: 876 / 6: 306 / 7: 256 / 8: 108. 최대 arity 8 → HAHE 입력 길이 15.
- HAHE는 문장의 모든 칸(주어·관계·목적어·한정 속성·한정값)을 하나씩 가려 학습한다: epoch당 학습 질의 152,399개, batch 512 → **298 step/epoch**, 300 epoch = 89,400 step.
- 전역 하이퍼그래프: HAHE 규칙대로 학습 문장의 앞쪽 (1 − 0.8) = 20%, 즉 5,203개 하이퍼엣지.
- 파라미터 **26,584,063**(전역 GAT 층·노드/하이퍼엣지 임베딩 6,742,528 포함). `native_hyper`(5,318,216)의 약 5.0배다.
- 평가는 HAHE 출력(어휘 전체 logit) 중 엔티티 구간만 꺼내 StarE id 순서로 놓고 위 규칙을 적용한다.

## 3. 예산과 운영

- 학습 설정: HAHE README의 WD50K 명령 그대로 — hidden 256, 전역 2층(GATv2, head 4, dropout 0.1, ELU), 지역 12층(head 4, dropout 0.1, GELU), batch 512, AdamW(lr 5e-4, weight decay 0.01), OneCycleLR(warm-up 10%, 선형), entity soft label 0.2, relation soft label 0.1, hyperedge dropout 0.8, **300 epoch**.
- `native_hyper`는 StarE 기본 예산(401 epoch)으로, HAHE는 자기 공개 예산(300 epoch)으로 판정한다. **epoch 수를 맞추지 않는다.** 두 모델의 학습 질의 구성이 달라 epoch이 같은 양의 학습을 뜻하지 않기 때문이다. 대신 학습 step 수·벽시계·파라미터 수를 결과 문서에 함께 적는다.
- 평가: 10 epoch마다와 300에서 test. **판정은 300 epoch 값.** test 최고값은 참고로만.
- seed `1 2 3 4 5`. HAHE 원본은 seed를 고정하지 않으므로, python·numpy·torch seed와 DataLoader 섞기 generator를 seed로 고정한다(추가한 것은 이것뿐).
- 운영: Study 3과 같다 — step당 150 ms GPU 휴지, 체크포인트 5 epoch(모델·optimizer·scheduler·RNG·섞기 상태), `LONG_RUN.lock`, fsync, 드라이버 `run_stare_judged.ps1 -Module src.ontology.hahe_bridge.runner`(완료 실행 건너뜀, 부분 실행 재개, 전력 한도 ≤ 160 W 대기, SM 클럭 > 1,450 MHz 두 번이면 중단). OneCycleLR가 전체 step 수에 묶여 있으므로, 다른 `--epochs`로 재개하려 하면 러너가 거부한다.
- 파일럿은 두지 않는다. 스모크(GPU, seed 0, 2 epoch + 중단·재개 1회)로 시간만 재고 §3-b에 적는다. 스모크의 test 지표는 읽지 않는다.

### 3-b. 스모크 실측 (2026-10-28 02:00, 판정 실행 전)

`outputs/hahe_same_harness_v1/smoke/hahe/seed_0/`, seed 0, 판정 split, `--epochs 2`. 1 epoch 뒤 `--stop-after-epochs 1`로 프로세스를 끝내고(중단 흉내), 같은 명령에 `--resume`을 붙여 epoch 2를 마쳤다.

| 항목 | 값 |
|---|---|
| epoch당 벽시계(휴지 포함) | 161.2 s / 160.8 s |
| 그중 휴지 | 44.5 s (297 × 150 ms) |
| 데이터 준비만(CPU, 따로 잼) | 약 13.5 s/epoch (0.045 s/batch) |
| test 평가 1회(10,594 질의, 두 채점 규칙) | 17.2 s / 16.6 s |
| peak VRAM | 4,023 MB |
| GPU 상태 | SM 1,395 MHz(사용자 상한 적용), 전력 한도 160 W, 실사용 약 75 W |
| 실행당 예상 | 300 × 161 + 30 × 17 ≈ 48,800 s ≈ **13.6 h** |

- 재개: `resumed_from_epoch = 1`, metrics.jsonl 1행 유지·0행 삭제, `summary.json`의 `epochs_done = 2`.
- 5 실행 합계 ≈ 68시간(약 2.8일). 시작 2026-10-28 새벽 기준 종료 예상 2026-10-30 밤. Windows Update 일시 중지는 2026-11-25까지다.
- 연산의 대부분은 GPU다(데이터 준비는 epoch의 8%). `--num-workers`는 0 그대로 둔다.
- 기록: 진행 감시 출력에 epoch 1의 test 줄(MRR 0.186)이 함께 찍혀 보였다. 1 epoch 값이라 아무 결정에도 쓰지 않았다. epoch 2 값은 읽지 않았다.
- 예산·규칙은 §3·§4 그대로다. 이 표는 시간 기록이며 아무것도 바꾸지 않는다.

판정 실행 명령(리셋 뒤에도 같은 명령):

```
powershell -File .harness/scripts/run_stare_judged.ps1 -Detach -Module src.ontology.hahe_bridge.runner -Arms hahe -Epochs 300 -OutRoot outputs\hahe_same_harness_v1\judged
```

### 3-c. 운영 기록: 앱 업데이트로 인한 중단 (2026-10-28)

- 08:41:44 seed 1이 epoch 152에서 멈췄다. 러너는 오류를 남기지 않았고, 드라이버도 종료 기록 없이 사라졌다. 재부팅은 없었다.
- 원인: Claude 데스크톱 앱 자동 업데이트(08:31 내려받기, 08:41:43~44 앱 재시작). 드라이버를 앱 안에서 `Start-Process`로 띄웠기 때문에 앱 프로세스 트리와 함께 종료됐다. 같은 초에 NVIDIA 드라이버 오류(nvlddmkm 이벤트 153)가 한 건 있다. 이것이 원인인지 결과인지는 1초 해상도로 구별할 수 없다.
- 조치: 드라이버의 `-Detach`가 WMI(`Win32_Process.Create`)로 드라이버를 만든다. 부모가 앱이 아니어서 앱 재시작의 영향을 받지 않는다(사용자 세션·GPU 접근은 시험으로 확인). 16:59에 같은 명령으로 재개했고, seed 1은 epoch 150 체크포인트에서 이어진다(151·152행은 재개 때 삭제).
- GPU가 약 8.3시간 놀았다. 종료 예상은 2026-10-31 새벽으로 늦어진다. 예산·규칙은 그대로다.

## 4. 판정 규칙 (동결, 양측)

규칙은 Study 2·3과 같다: test **MRR과 Hits@10**, 각 arm의 마지막 epoch. seed마다 엄격히 큰 쪽이 이긴다. 두 지표 모두 5 seed 중 4 이상을 이긴 쪽의 이름이 verdict이고, 아니면 `no_consistent_order`. 동점은 승이 아니다. 방향은 미리 정하지 않는다.

| 판정 | 비교 | 질문 |
|---|---|---|
| **J3** | `native_hyper`(Study 3, 401 epoch) vs `hahe`(이 실험, 300 epoch) | 같은 데이터·질의·후보·채점에서 우리 인코더가 HAHE보다 위인가 아래인가 |

- `native_hyper` 값은 이미 관측됐다(0.6648 ± 0.0026). 이 사전등록이 묶는 것은 HAHE 쪽 실행과 판정 규칙이다.
- seed 번호로 짝을 짓지만 두 모델은 코드가 달라 seed가 같은 난수를 뜻하지 않는다. 쌍 차이 통계는 참고용이고, 판정은 seed별 승수로만 한다.

보고(판정 아님):

- **이식 확인**: HAHE 자체 규칙(모든 엔티티 후보, 동점 불리)의 주어·목적어 MRR 평균을 GLoRE 재실행값 0.670과 비교한다. ±0.03 안이면 "이식이 문헌 값을 재현"으로 적는다. 벗어나면 그 사실만 적고(§6), 판정은 그대로 둔다. 우리 필터 키는 한정쌍을 원래 순서로 쓰고 HAHE는 정렬해 쓰므로, 이 값은 HAHE 원본 평가와 완전히 같지는 않다.
- seed별 표, Hits@1·Hits@3·MR, 목적어/주어 예측 분리, 곡선, 학습 step·벽시계·파라미터.

판정 명령:

```
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other hahe --other-root outputs/hahe_same_harness_v1/judged --other-epochs 300
```

## 5. 사전 예측 (틀릴 수 있게)

- **J3**: `hahe_wins` 50%, `no_consistent_order` 30%, `native_hyper_wins` 20%. 근거: GLoRE 재실행에서 HAHE(0.670)가 우리 값(0.6648)보다 0.005 위였고, HAHE는 파라미터가 5배이며 12층 Transformer로 문장 안 위치를 직접 본다. 반대 근거: 탐색 없이 WD50K 설정을 그대로 쓰고, 그 설정은 문장이 더 많은 WD50K 전체용이다.
- **StarE 규칙 MRR**: 0.64~0.70 (중앙 0.67).
- **이식 확인**: HAHE 자체 규칙 MRR은 StarE 규칙보다 낮게 나온다(후보가 18,791로 늘어 순위가 같거나 나빠지므로). 0.670 ± 0.03 안일 확률 65%.

## 6. 읽기 규칙 (관측 전 고정)

| 결과 | 읽기 |
|---|---|
| J3 `hahe_wins` | 논문 위치 문장을 "같은 하네스에서 HAHE보다 낮다"로 바꾼다. 통제 비교(J1·J2)의 결론은 그대로이고, 우리 인코더는 경쟁 모델이 아니라 통제 비교용이라고 적는다 |
| J3 `no_consistent_order` | "같은 하네스에서 HAHE와 구별되지 않는다"로 적는다 |
| J3 `native_hyper_wins` | "같은 하네스·HAHE 공개 설정 기준으로 HAHE보다 높다"로 적되, HAHE에 탐색을 하지 않았다는 한계를 같은 문장에 붙인다. SOTA 주장은 하지 않는다 |
| 이식 확인 실패 | 우리 이식 또는 프로토콜 차이로 HAHE가 문헌 값만큼 나오지 않았다고 적는다. GLoRE 그림 값도 표에 그대로 남긴다. 버그 사냥 재실행은 하지 않고, 필요하면 새 charter를 쓴다 |

어느 경우든 GLoRE 그림에서 읽은 값(HAHE ≈ 0.670, MAYPL ≈ 0.658, GLoRE ≈ 0.824)은 출처를 밝혀 표에 남기고, 같은 하네스 값은 별도 행으로 넣는다. arm·예산·규칙은 결과를 본 뒤 바꾸지 않는다.

## 7. Non-goals

- HAHE 하이퍼파라미터 탐색 없음. epoch을 401로 맞춘 추가 실행 없음.
- HAHE 원본 코드·HAHE 저장소 데이터 파일 사용·수정 없음(`src/*.py`만 import).
- 관계·한정 속성·한정값 예측 평가 없음. HAHE는 그것들도 학습하지만(원본 그대로) 판정·보고는 주어·목적어만 한다.
- MAYPL·GLoRE 실행 없음(후속 후보). 다른 데이터셋 없음.
- Study 1~3의 수치·verdict 수정 없음.

## 8. Touched files

- 신규: `src/ontology/hahe_bridge/{__init__,data,upstream,runner}.py`, `tests/test_hahe_bridge.py`(11), `.harness/scripts/fetch_hahe.py`, `third_party/hahe.lock.json`, 이 charter, 결과 `docs/hahe-same-harness-v1-results.md`.
- 수정: `.harness/scripts/run_stare_judged.ps1`(`-Module`, 기본값이면 동작 불변), `.harness/scripts/judge_stare_input_representation.py`(`--other-root`·`--other-epochs`, 기본값이면 출력 불변), `tests/test_stare_judged.py`(1개 추가), `.gitignore`(clone·새 출력의 체크포인트·로그·어휘 파일).
- 판정 뒤 갱신: `papers/preserve_vs_reify/preserve_vs_reify.tex` Table 5·위치 문장·한계, `docs/related-work-hkg-comparison.md` §2-a, `docs/charter.md`, `docs/checkpoints.md`, 이벤트 로그.
- 산출물: `outputs/hahe_same_harness_v1/{smoke,judged}/hahe/seed_{k}/`.

## 9. Required checkpoints

`charter_frozen` → `hahe_contract`(테스트 + Ruff + lock 검증) → `smoke`(GPU, 중단·재개 1회) → `budget_addendum`(§3-b) → `judged_run`(5 실행) → `judged_verdict`(J3 + 이식 확인) → `results_doc` → `paper_update`(수치 검사·PDF 빌드) → `checkpoint_gate` → `clarification_debt`.

## 10. Rollback

신규 파일 삭제, 드라이버·판정 스크립트의 선택 인자 되돌림(기본 동작은 이미 불변), `third_party/HAHE` 삭제. 기존 Study 1~3 산출물은 읽기만 한다.
