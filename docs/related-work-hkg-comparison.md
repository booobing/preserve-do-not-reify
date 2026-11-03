# 선행 Hyper-relational KG 모델과의 비교 (2026-10-21)

**2026-10-27 갱신**: §6의 1번 실험을 마쳤다(`docs/stare-native-encoder-v1-results.md`). 우리 5장 인코더가 이제 §2-a 표에 들어가며, §한 줄 결론 2번·§4·§6을 그에 맞게 고쳤다. 원래 문장은 지우지 않고 갱신 표시를 붙였다.

`papers/`에 넣은 선행 논문 13편을 원문에서 직접 읽어 정리했다. 모든 수치는 PDF의 표에서 옮겼고, 표 번호와 쪽을 적었다.
차트에서 눈으로 읽은 값은 "차트 판독"이라고 따로 표시했다. 기억이나 2차 자료에서 가져온 수치는 없다.

## 이 문서에서 쓰는 용어

- **HKG(hyper-relational knowledge graph)**: 주 사실 (주어, 관계, 목적어)에 (한정 속성, 한정값) 쌍이 붙는 지식 그래프. Wikidata의 qualifier가 대표 예다.
- **한정쌍을 다루는 방식**: 이 비교의 축이다.
  - *엣지 특징*: 한정쌍을 주 사실 엣지의 관계 벡터에 섞는다. 한정값 엔티티는 메시지를 만들 때만 쓰이고 그 사실로부터 메시지를 받지 않는다(StarE).
  - *참여자(하이퍼엣지 구성원)*: 한정값도 사실의 구성원으로 두고 사실을 거쳐 메시지를 주고받는다(HAHE, GLoRE, KREPE, 우리 5장 인코더).
  - *서열 토큰*: 사실을 `[s, r, o, k₁, v₁, …]` 토큰 열로 보고 Transformer로 읽는다(GRAN, HyperFormer, HAHE의 지역 부분).
  - *분해*: 사실을 2항 엣지로 바꾼다. reification(사실 노드 + 역할 엣지), 쌍 분해(star-to-clique), 한정쌍 버리기(prune) 등.
- **main / all 프로토콜**: "main"(또는 s/o, Pri)은 주 사실의 주어·목적어만 가리고 맞힌다. "all"은 한정값 자리까지 가린다. 같은 데이터라도 두 프로토콜의 수치는 나란히 놓을 수 없다.
- **WD50K / WD50K(100)**: StarE가 만든 Wikidata 기반 벤치마크. WD50K는 사실 236,507개 중 13.6%에 한정쌍이 있고, WD50K(100)은 모든 사실(31,314개)에 한정쌍이 있는 부분집합이다. 우리 실험은 전부 WD50K(100)에서 했다.

## 한 줄 결론

1. **우리 StarE 이식 수치(0.6553)는 문헌 표에 바로 놓인다.** WD50K(100) main 프로토콜에서 StarE 원 논문 0.654를 재현했고, 같은 백본의 reified 그래프는 0.6485였다.
   그러나 이 데이터에서 최신 모델(HAHE·MAYPL·GLoRE)은 GLoRE 발표 자료의 차트로만 확인되며, GLoRE는 약 0.83(차트 판독)으로 StarE보다 훨씬 높다.
2. **우리 5장 인코더의 수치(0.147)는 어느 문헌 표에도 들어가지 않는다.** 가리는 위치·후보 집합·예산·채점이 모두 달라서다. 이 인코더를 기존 모델과 비교하려면 StarE 프로토콜로 다시 돌려야 한다(§6).
   **(2026-10-27 갱신)** 표준 프로토콜로 다시 돌린 결과 우리 보존 인코더는 WD50K(100) main MRR **0.6648 ± 0.0026**(5 seed)으로 §2-a 표에 들어간다. StarE 원본(0.6553)보다 조금 높고 HAHE·MAYPL(약 0.68)·GLoRE(약 0.83)보다 낮다.
   **(2026-10-31 갱신)** HAHE 공개 코드를 같은 하네스(같은 행·질의·후보·채점)로 직접 돌려 판정했다: HAHE 0.6784 대 우리 0.6648, MRR·Hits@10 모두 5/5로 HAHE가 높다(`docs/hahe-same-harness-v1-results.md`). 우리 인코더는 같은 프로토콜에서 HAHE보다 아래다.
3. **"보존 대 분해"를 같은 모델로 비교한 선행 연구는 세 편이고, 결론이 엇갈린다.** m-TransH(2016)는 사실 ID 노드를 더한 쪽이 더 좋았고, HINGE(2020)는 2항 모델에 reification을 줘도 HINGE를 못 이겼으며,
   "Understanding"(2025)은 한정쌍을 아예 버린 NBFNet이 HKG 전용 모델과 거의 같다고 보고했다. 우리 5장과 StarE 실험은 **같은 인코더·같은 파라미터·같은 예산에서 표준 reification을 대조군으로 둔** 비교라는 점에서 이들과 다르다.

## 1. 모델 지형 — 한정쌍을 어떻게 다루는가

| 모델 | 출처(원문 표기) | 한정쌍 처리 | 한정값이 메시지를 받나 | 코드 |
|---|---|---|---|---|
| m-TransH | Wen et al., arXiv 1604.08642; IJCAI 2016(NaLP 참고문헌 표기) | 역할→엔티티 함수(instance), 역할마다 실수 가중치 | 해당 없음(점수 함수) | 미기재 |
| NaLP / tNaLP⁺ | Guan et al., 저널 확장판 arXiv 2104.10424(원 학회판 WWW 2019) | 역할:값 쌍 집합, 모든 쌍을 대칭으로 비교 | 해당 없음(점수 함수) | github.com/gsp2014/NaLP |
| HINGE | Rosso et al., WWW '20, "Beyond Triplets" | 주 사실 3×K 행렬 + 쌍마다 5×K 행렬 CNN | 아니다(메시지 전달 없음) | github.com/eXascaleInfolab/HINGE_code |
| StarE | Galkin et al., arXiv 2009.10847; EMNLP 2020 | **엣지 특징**: φq로 합성·합산 뒤 관계와 가중합(α = 0.8) | **아니다** | github.com/migalkin/StarE |
| GRAN | ACL Findings 2021(GLoRE 자료 표기) | 서열 토큰 + 엣지 편향 어텐션 | 서열 안에서만 | 원문 미보유 |
| HyperFormer | CIKM 2023(GLoRE 자료 표기) | 서열 + 1-hop 이웃 집계 | 서열 안에서만 | 원문 미보유 |
| HAHE | Luo et al., arXiv 2305.06588; ACL 2023(GLoRE 자료 표기) | **참여자**(전역 하이퍼그래프) + 서열(지역) | **그렇다** | github.com/LHRLAB/HAHE |
| FormerGNN | Wang et al., arXiv 2508.03280 | 서열 토큰(HyperFormer) + 한정쌍 뺀 그래프의 NBFNet | 그래프 쪽은 아니다 | github.com/Wyb0627/GraphEmbedding |
| MAYPL | ICML 2025(GLoRE 자료 표기) | 한정쌍이 유도한 구조 위 메시지 전달 | 원문 미보유 | 원문 미보유 |
| KREPE | Lee et al., ICML 2026(PMLR 306), arXiv 2605.24064 | **참여자**: 관계-엔티티 쌍의 합을 사실 표현으로, 마스크 이산 확산 학습 | **그렇다** | github.com/bdi-lab/KREPE |
| GLoRE | Ji et al., KDD 2026(KDD 2026 Proceedings V.2, pp. 2050–2061, doi:10.1145/3770855.3818000; 본문 접근 불가로 수치는 2026-07-29 발표 자료) | **참여자**: 사실 = 엔티티·관계 노드의 하이퍼엣지, 관계-엔티티 쌍 단위 전파 | **그렇다** | 미기재 |
| HEHRGNN | Rajagopalamenon & Cheramangalath, arXiv 2602.18897 | 참여자: 사실을 하이퍼엣지 허브로, 주/한정 노드별 가중치 | 그렇다 | github.com/gitrajeshr/HEHRGNN |
| **우리 5장 인코더** | 이 저장소 | **참여자**: 역할 임베딩을 단 하이퍼엣지, 노드→엣지→노드 대칭 집계 | **그렇다** | `src/ontology/encoder.py` |

비교 대상이 아닌 세 편: ÆMP(KBS 2022)는 2항 KG의 관계 예측, MKGEM(ICDSCNC 2024)은 2항 멀티모달 추천이라 한정쌍이 없다.
HKGNN(arXiv 2311.16683, 다음 POI 추천)은 역할 없는 n항 튜플을 clique로 펴서 GAT에 넣는 **응용 사례**이며, 리뷰가 요구한 "응용 맥락"의 참고로만 쓸 수 있다.

**우리 인코더의 위치**: 한정값을 참여자로 두고 대칭으로 집계한다는 점에서 StarE가 아니라 HAHE·GLoRE·KREPE 쪽에 속한다. GLoRE 자료(p31)의 분류로는 "전역(global)만, 노드 단위"이며,
GLoRE는 같은 노드 단위 전역 집계를 쓰는 변형(NODE)이 WD50K main MRR 0.348로 쌍 단위(0.478)보다 크게 낮다고 보고했다(p56 Table 9). 이것은 우리 인코더의 설계에 대한 직접적인 경고다(§6).

## 2. 같은 자리에 놓을 수 있는 수치

### 2-a. WD50K(100), main 프로토콜 (주어·목적어 예측, filtered, test)

| 모델 | MRR | Hits@1 | Hits@10 | 출처 |
|---|---|---|---|---|
| Transformer (T) — 한정쌍 없음 | 0.351 | 0.261 | 0.530 | StarE 논문 Table 3, p7 |
| StarE (T) + Transformer (T) — 한정쌍 없음 | 0.380 | 0.276 | 0.584 | 같은 표 |
| NaLP-Fix | 0.458 | 0.398 | 0.563 | 같은 표 |
| HINGE | 0.492 | 0.417 | 0.636 | 같은 표 |
| Transformer (H) | 0.562 | 0.499 | 0.677 | 같은 표 |
| StarE (H) + Transformer (H) | 0.654 | 0.588 | 0.777 | 같은 표 |
| **우리 이식: StarE 원본 그래프** (5 seed, 401 epoch) | **0.6553 ± 0.0013** | 0.5891 | 0.7766 | `docs/stare-input-representation-v1-results.md` |
| **우리 이식: reified 그래프** (5 seed, 401 epoch) | **0.6485 ± 0.0021** | 0.5809 | 0.7708 | 같은 문서 |
| **우리 5장 인코더, 한정쌍 제거** (5 seed, 401 epoch) | 0.6519 ± 0.0026 | 0.5871 | 0.7709 | `docs/stare-native-encoder-v1-results.md` |
| **우리 5장 인코더, reification** (5 seed, 401 epoch) | 0.6561 ± 0.0024 | 0.5942 | 0.7686 | 같은 문서 |
| **우리 5장 인코더, 하이퍼엣지 보존** (5 seed, 401 epoch) | **0.6648 ± 0.0026** | **0.6025** | **0.7777** | 같은 문서 |
| **HAHE, 같은 하네스로 우리가 실행** (5 seed, 300 epoch, 공개 설정·탐색 없음) | **0.6784 ± 0.0017** | **0.6152** | **0.7947** | `docs/hahe-same-harness-v1-results.md` |
| MAYPL (GLoRE 저자 재실행) | 0.658 (그림 판독) | — | — | Ji et al. KDD 2026 논문 Figure 5(a) |
| HAHE (GLoRE 저자 재실행) | 0.670 (그림 판독) | — | — | 같은 그림 |
| GLoRE | 0.824 (그림 판독) | — | — | 같은 그림 |

- StarE 원 논문 표에서 한정쌍을 쓰는 모델(H)과 안 쓰는 모델(T)의 차이가 WD50K(100)에서 가장 크다(0.654 대 0.380). 이 데이터는 한정쌍이 모든 사실에 있어 한정쌍의 효과를 보기에 가장 유리한 자리다.
- GLoRE 자료는 WD50K(100)에서 HAHE·MAYPL의 개별 수치를 표로 주지 않는다. 차트와 증가분 표기만 있으므로 논문에는 GLoRE 원 논문이 공개되면 그 표로 바꿔야 한다.
- **2026-10-28 갱신(논문 PDF 확보)**: GLoRE 논문(KDD 2026, doi:10.1145/3770855.3818000)도 WD50K(100)을 표가 아니라 Figure 5로만 보고한다. HAHE·MAYPL은 원 논문에 WD50K(33/66/100) 결과가 없어 GLoRE 저자들이 같은 탐색 절차로 재실행한 값이며, WD50K(100)의 최강 기준선은 HAHE다. 위 세 값은 Figure 5(a)(주 엔티티)를 400 dpi로 렌더링해 격자선 대비 표식 중심을 읽은 것이다. 읽은 값의 GLoRE − HAHE = +0.1538(23.0%)로 그림에 인쇄된 +0.153(+23%)과 0.001 안에서 맞고, 판독 오차는 약 ±0.003으로 본다.
- **2026-10-31 갱신(HAHE 직접 실행)**: HAHE 공개 코드(commit `e35e0a7b`, 수정 없음, README의 WD50K 설정, 300 epoch)를 StarE 하네스의 행·질의·후보·채점식 그대로 돌렸다. 판정 J3 `hahe_wins`: MRR·Hits@10 모두 5/5 seed에서 HAHE가 높고, MRR 차이는 +0.0135다.
  HAHE 자체 규칙(엔티티 전부가 후보, 동점은 불리하게)으로는 0.6778이라 GLoRE 재실행값(약 0.670)과 맞는다. 그림 판독으로는 우리 인코더가 MAYPL과 HAHE 사이였지만, 같은 하네스에서는 HAHE보다 0.0135 아래다.
  우위는 주어 예측에서만 나왔다(+0.0305, 5/5). 목적어 예측 MRR은 다섯 seed 중 넷에서 우리가 높다. MAYPL·GLoRE는 여전히 그림 판독값이다.

### 2-b. WD50K 전체, main / all 프로토콜

| 모델 | main MRR / H@1 / H@10 | all MRR / H@1 / H@10 | 출처 |
|---|---|---|---|
| NaLP-Fix | 0.177 / 0.131 / 0.264 | 0.224 / 0.158 / 0.330 | GLoRE 자료 p52 |
| HINGE | 0.243 / 0.176 / 0.377 | 0.232 / 0.164 / 0.343 | 같은 표 |
| StarE | 0.349 / 0.271 / 0.496 | — | StarE 논문 Table 3 |
| ShrinkE (2023) | 0.345 / 0.275 / 0.482 | — | KREPE Table 1, p6 |
| Hy-Transformer (2021) | 0.356 / 0.281 / 0.498 | — | 같은 표 |
| HyNT (2023) | 0.357 / 0.281 / 0.501 | 0.383 / 0.308 / 0.527 | 같은 표 |
| NBFNet, 한정쌍 버림(Prune) | 0.360 / 0.271 / 0.530 | — | Understanding Table 3, p5 |
| HyperFormer (2023) | 0.366 / 0.288 / 0.514 | — | KREPE Table 1 |
| HAHE (2023) | 0.368 / 0.291 / 0.516 | 0.402 / 0.327 / 0.546 | HAHE Table 2, p7 |
| HDiff (2025) | 0.373 / 0.292 / 0.512 | 0.405 / 0.325 / 0.546 | KREPE Table 1 |
| FormerGNN (2025) | 0.377 / 0.299 / 0.527 | — | 같은 표 |
| MAYPL (2025) | 0.381 / 0.297 / 0.544 | 0.411 / 0.326 / 0.572 | 같은 표 |
| KREPE (2026) | 0.389 / 0.303 / 0.552 | 0.419 / 0.334 / 0.580 | 같은 표 |
| GLoRE (2026) | 0.478 / 0.370 / 0.684 | 0.512 / 0.401 / 0.718 | GLoRE 자료 p52 |

- 이 표가 현재 HKG 분야의 실제 경쟁 지형이다. 2020~2025년 모델들은 main MRR 0.35~0.39에 몰려 있고, GLoRE만 0.478로 크게 벗어난다.
- **한정쌍을 버린 NBFNet(0.360)이 StarE(0.349)보다 높다.** "Understanding" 논문의 핵심 주장이며, "한정쌍을 구조로 쓰면 낫다"는 우리 주장에 가장 불리한 선행 근거다(§3).
- 우리는 WD50K 전체를 돌린 적이 없다. StarE 논문 기준 이 데이터는 WD50K(100)의 7.5배이고 학습에 V100으로 약 2일이 걸렸다(StarE App. C).

### 2-c. JF17K, WikiPeople

수치는 StarE Table 2, HAHE Table 2, NaLP Table 2·6, HINGE Table 2, GLoRE 자료 p52, KREPE Table 2에 있다. 두 가지 이유로 이 문서의 비교 축에서 뺐다.
JF17K는 test 사실의 약 44.5%가 train 사실과 (s, r, o)를 공유하는 누수가 있다(StarE 논문 4.1절, 정리 뒤 StarE MRR 0.574 → 0.376).
WikiPeople은 한정쌍이 있는 사실이 2.6%뿐이라 한정쌍 표현의 차이가 드러나기 어렵다. 리뷰 ①이 요구한 두 번째 데이터셋으로는 WD50K(33)·(66)이 같은 코드로 바로 돌고 한정쌍 비율만 달라 더 깨끗한 대조가 된다.

### 2-d. 우리 5장 수치 0.147이 이 표들에 없는 이유

| 항목 | 우리 5장 V2-D | 문헌 main 프로토콜 |
|---|---|---|
| 가리는 위치 | 모든 역할 중 하나를 무작위로(한정값 포함) | 주어·목적어만 |
| 후보 | 로더의 엔티티 18,025개 전부(StarE 로더 기준 모든 자리 18,791개) | 한정값으로만 나온 7,917개를 뺀 10,874개 |
| 예산 | 1,500 step × batch 32, d = 64 | 400 epoch 안팎, d = 200 |
| 채점 | test 2,000개 표본, forward 하나에 128개 동시 마스크 | test 전부, 질의마다 따로 |
| 그래프 | 평가 때 전체 그래프(가린 멤버만 제외) | train(+valid) 사실만 |

"all" 프로토콜과도 다르다. all은 한정값 자리를 포함하지만 후보 집합·예산·채점이 여전히 다르다. **우리 인코더의 절대 성능은 지금 어떤 문헌 수치와도 비교할 수 없다.**

## 3. "보존 대 분해" 선행 비교

| 연구 | 같은 모델로 비교했나 | 무엇을 비교했나 | 결과 |
|---|---|---|---|
| m-TransH (2016), JF17K | 그렇다 | 쌍 분해(TransH, S2C) / 직접 n항(m-TransH) / n항 + 사실 ID 역할(m-TransH:ID) | HIT@10(DIM 250): 49.26% / 68.73% / **80.82%** (Table 4, p7). n항이 쌍 분해보다 낫고, **사실 노드를 더한 쪽이 가장 좋다** |
| HINGE (2020), JF17K·WikiPeople | 아니다(2항 모델에 분해 입력, HINGE는 원 사실) | 2항 모델 8종 × Basic / Relation Path / Reification vs HINGE | 2항 모델은 Reification이 Basic보다 나쁘다(WikiPeople H/T 평균 −2.2%). HINGE는 최고 2항 설정보다 JF17K H/T에서 29.71% 높다(Table 3, p9) |
| Understanding (2025), Cleaned JF17K·WD50K | 그렇다 | NBFNet·CompGCN에 Prune / Direct / Hyper 분해 | NBFNet은 한정쌍을 더할수록 나빠진다(WD50K 0.360 → 0.351 → 0.346). CompGCN은 거의 같다(0.348 → 0.350). 표준 reification(사실 노드)은 **돌리지 않았다** |
| StarE (2020), WD50K 계열 | 그렇다 | 같은 구조에서 한정쌍 사용(H) / 미사용(T) | WD50K(100) 0.654 대 0.380. reification 대조는 없다 |
| GLoRE (2026), WD50K 등 | 그렇다 | 노드 단위 전역 집계 vs 관계-엔티티 쌍 단위 집계 | WD50K main 0.348 대 0.478(p56 Table 9). 역할 혼동: 노드 단위에서 주어·목적어의 74.8%를 한정값으로 오인(p53) |
| KREPE (2026), WikiPeople⁻ | 아니다 | 사실을 토큰 열로 직렬화한 BERT·BART vs KREPE | 링크 예측 MRR 0.503 / 0.477 / 0.522(Table 12, p15) |
| **우리 5장** (2026-08) | **그렇다(같은 모듈·파라미터·초기 가중치·예산)** | 하이퍼엣지 보존 vs 표준 reification(빈 노드 + 역할 엣지), 4개 통제 + 깊이 steelman | 7개 조건 모두 5/5. WD50K(100)(우리 프로토콜) MRR 0.147 대 0.118 |
| **우리 StarE 실험** (2026-10) | **그렇다(StarE 코드 무수정, 그래프만 교체)** | StarE 원본(엣지 특징) vs 표준 reification 그래프 | 401 epoch 5/5 원본 우세, 0.6553 대 0.6485. 76 epoch에서는 반대(0.434 대 0.336) |

**읽는 법.**

1. 우리 비교의 새로운 점은 **대조군이 표준 reification(사실 노드 + 역할 엣지)이고 나머지를 모두 같게 맞췄다**는 데 있다. 선행 연구는 대조군이 쌍 분해(m-TransH, Understanding의 Direct/Hyper)이거나,
   모델이 서로 다르거나(HINGE), reification을 그림으로만 다뤘다(Understanding Fig. 1(d), GLoRE 자료 p10의 Fatemi et al. 인용).
2. **m-TransH의 사실 ID 결과는 우리 주장과 반대 방향이다.** 사실 노드를 더하자 HIT@10이 크게 올랐다. 다만 그 실험의 "+ID"는 n항 사실을 유지한 채 노드를 **추가**한 것이고,
   우리의 reification은 n항 사실을 노드와 2항 엣지로 **대체**한 것이라 같은 처치가 아니다. 논문에서 이 차이를 명시해야 한다.
3. **"Understanding"은 "한정쌍을 구조로 넣으면 낫다" 자체를 흔든다.** 한정쌍을 버린 NBFNet이 HKG 전용 모델과 비슷했다. 우리 결과는 "어떻게 넣느냐" 사이의 비교라 이 주장과 직접 충돌하지는 않지만,
   한정쌍을 버린 대조군(Prune)이 우리 실험에 없다는 것은 심사에서 지적받을 수 있다.
4. **GLoRE의 노드 대 쌍 결과는 우리 인코더에 대한 경고다.** 우리 인코더는 역할 임베딩을 메시지에 더하므로 GLoRE의 "역할 무시 노드 집계"와 같지는 않다. 그러나 엔티티당 표현이 하나라는 점은 같고,
   GLoRE는 이것이 역할 혼동을 만든다고 보고했다. 우리 인코더를 StarE 프로토콜로 돌리면 이 차이가 수치로 드러날 것이다.

## 4. 우리 기여의 위치 (논문에 쓸 문장의 근거)

- **주장할 수 있는 것**: 정보·모듈·파라미터·초기 가중치·예산을 같게 맞춘 조건에서, N항 사실을 하이퍼엣지로 보존한 표현이 표준 reification보다 N항 보완에서 일관되게 낫다(우리 인코더 7조건 5/5).
  **(2026-10-27 추가)** 같은 5장 인코더를 표준 프로토콜·표준 예산(StarE 기본값 401 epoch)으로 돌려도 보존이 reification(5/5, MRR −0.0087)과 한정쌍 제거(5/5, MRR −0.0129)를 모두 이겼다. 5장 프로토콜, StarE 백본에 이은 세 번째 확인이다.
  공개 모델 StarE의 코드를 그대로 쓴 두 번째 백본에서도 수렴 예산에서는 같은 방향이었다(5/5, 격차 MRR 0.007). 이 조합의 통제 비교는 읽은 13편에 없다.
- **주장할 수 없는 것**:
  - 기존 HKG 모델보다 성능이 좋다 — **(2026-10-27 갱신)** 표준 프로토콜에서 0.6648로 StarE(0.6553)보다 조금 높지만 파라미터가 4.9% 많고, HAHE·MAYPL(약 0.68)·GLoRE(약 0.83)에는 못 미친다.
    **(2026-10-31 갱신)** 같은 하네스에서 HAHE에 5/5로 졌다(0.6784 대 0.6648). "HAHE보다 낮다"는 이제 그림 판독이 아니라 판정 결과다.
  - 한정쌍을 구조로 넣는 것이 필수다 — Prune 대조군이 없고, "Understanding"은 반대 근거를 낸다.
    **(2026-10-27 갱신)** Prune 대조군을 돌렸다: WD50K(100)에서 보존이 한정쌍 제거를 5/5로 이겼다(MRR −0.0129, 디코더는 두 arm 모두 한정쌍 토큰을 봄). 이 데이터에서는 "버려도 비슷하다"가 맞지 않았다. 다만 WD50K 전체(한정쌍 13.6%)와 다른 모델에서는 재지 않았으므로 "필수"라고는 쓰지 않는다.
  - 사실 노드는 언제나 해롭다 — m-TransH는 사실 노드를 **추가**했을 때 좋아졌다.

## 5. 인용 시 주의 — 원문에서 확인한 오류와 불일치

| 대상 | 문제 | 처리 |
|---|---|---|
| HEHRGNN Table 4 | WD50K MRR 0.759는 같은 행의 Hits@1/3/10으로 가능한 최대치(약 0.716)를 넘는다. Combined-HEHR 0.612도 최대치(약 0.487)를 넘는다. filtered 여부·후보 집합 미기재, StarE와 프로토콜이 다르다고 스스로 밝힘 | **인용하지 않는다** |
| t-HINGE Table II | NaLP WikiPeople 행의 열이 밀렸다(H@3를 H@10 자리에, H@10을 MRR 자리에). m-TransH·RAE 행은 MRR = Hits@1 | 원 논문 수치를 쓴다 |
| HAHE Table 2 | NeuInfer의 WD50K s/o 행이 HINGE 수치와 같다. StarE의 WikiPeople 관계 예측 행이 엔티티 예측 행과 같다 | 해당 행 인용 안 함 |
| Understanding Table 6 | StarE의 FBAUTO 행이 WD50K 행과 같다. seed·분산 미보고. StarE를 0.346, HAHE를 0.360으로 재측정(원 논문 0.349, 0.368) | 재측정값임을 밝혀 인용 |
| 데이터셋 통계 | WikiPeople 관계 수가 논문마다 707 / 375 / 193 / 178 / 171로 다르다. JF17K 관계 수도 322 / 327 / 501 | 비교 표에 관계 수를 쓰지 않는다 |
| JF17K | test의 약 44.5%가 train과 (s, r, o) 공유(StarE 논문) | 주 비교에서 뺀다 |
| 처음 받은 목록 | HINGE는 WWW 2020 "Beyond Triplets…"다(2021 "A Framework…"가 아님). HAHE의 원문 제목은 "…in Global and Local Level"이고 arXiv 2305.06588이다(WWW 2022 아님, 발표 자료는 ACL'23로 표기). GRAN은 ACL Findings 2021로 표기된다(EMNLP 2021 아님). m-TransH는 IJCAI 2016이다. 폴더의 NaLP는 학회판이 아니라 저널 확장판(arXiv 2104.10424)이다. "Hyper-SAG2V"는 폴더의 어느 논문에도 없어 확인하지 못했다 | 이 문서의 표기를 쓴다 |

## 6. 비교를 완성하려면 필요한 실험

우리 인코더를 기존 모델과 같은 표에 놓으려면 표준 프로토콜로 돌리는 것 말고는 방법이 없다. 우선순위 순으로 적는다. 셋 다 새 charter가 필요하다.

1. **[완료 2026-10-27: J1·J2 모두 `native_hyper_wins` 5/5, `docs/stare-native-encoder-v1-results.md`] 우리 5장 인코더를 StarE 프로토콜로 (WD50K(100), main, test, 5 seed).** StarE 이식 하네스(`src/ontology/stare_bridge/`)는 데이터 로더·sampler·filtered 평가·판정 스크립트를 이미 갖췄으므로,
   인코더만 우리 것으로 바꾸고 StarE 디코더를 그대로 쓰면 된다. 표현 arm은 셋으로 둔다: 우리 하이퍼엣지 보존, 같은 인코더의 reified 그래프, 한정쌍을 버린 Prune.
   이 한 실험으로 (a) 우리 인코더의 절대 수치를 §2-a 표에 놓고, (b) StarE 실험에서 배제 못 한 "엣지 특징 대 참여자" 설명을 가르고, (c) "Understanding"의 Prune 반론에 답한다.
   비용은 스모크로 잰다. 참고로 StarE 이식은 실행당 약 11시간(이 기계, 휴지 포함)이었다.
2. **[완료 2026-10-31: J3 `hahe_wins` 5/5, `docs/hahe-same-harness-v1-results.md`] HAHE 공개 코드를 WD50K(100)에서 (main, 5 seed).** 이 데이터에서 StarE보다 강한 기준선의 표 수치가 문헌에 없다(GLoRE 차트뿐). HAHE는 우리처럼 한정값을 참여자로 두는 대표 모델이고,
   원 논문 기준 11 GB GPU에서 WD50K 전체 8시간이라 WD50K(100)은 이 기계에서 감당할 만하다. KREPE는 2,000 epoch라 비용이 크다.
3. **WD50K(33)·(66)로 한정쌍 비율을 바꾸기.** StarE 하네스가 그대로 받는다. "보존의 이점이 한정쌍 비율에 따라 커지는가"를 보는 깨끗한 두 번째 축이며, 리뷰 ①의 "두 번째 데이터셋"을 대신할 수 있다.

WD50K 전체는 StarE 기준 V100 약 2일이라 이 기계에서 5 seed를 돌리기 어렵다. 다른 GPU가 생기면 1번을 WD50K 전체로 넓히는 것이 문헌과 가장 직접 맞닿는 비교다.

## 원문 위치

| 파일(papers/) | 실제 논문 |
|---|---|
| Message Passing for Hyper-Relational Knowledge Graphs.pdf | StarE |
| Beyond Triplets Hyper-Relational Knowledge Graph Embedding for Link Prediction.pdf | HINGE |
| Understanding the Embedding Models on Hyper-relational.pdf | FormerGNN 분석 논문 |
| HAHE- Hierarchical Attention for Hyper-Relational Knowledge Graphs.pdf | HAHE |
| Link Prediction on N-ary Relational Data.pdf | NaLP 저널 확장판 |
| On the Representation and Embedding of Knowledge Bases with General Relations.pdf | m-TransH("…Beyond Binary Relations") |
| Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion.pdf | KREPE |
| Hyper-Relational_Knowledge_Graph_Embedding_Based_on_Type_Constraints.pdf | t-HINGE |
| HEHRGNN- A Unified Embedding Model for Knowledge.pdf | HEHRGNN |
| [260729][지상준] Hyper-relational Knowledge Graphs and Beyond….pdf | GLoRE(KDD'26) 발표 자료 + HyperGraphRAG·Graph-R1 요약 |
| Hyper-Relational Knowledge Graph Neural Network for Next POI Recommendation.pdf | HKGNN(응용) |
| Learning attention-based representations from multiple patterns….pdf | ÆMP(2항 KG, 비교 대상 아님) |
| Multi-Modal_Knowledge_Graph_System_with_Embedding_Module.pdf | MKGEM(2항, 비교 대상 아님) |
