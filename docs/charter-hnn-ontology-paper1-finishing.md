# Charter — Paper 1 마무리: 참고문헌 · 영문 본문 · 통계 검정

작성 2026-09-13. 선행: `docs/hnn-ontology-paper1-draft.md` 부록 A의 남은 세 항목.
**통계 절(§3)은 검정을 계산하기 전에 쓰고 커밋 대상에 포함한다.** 수치를 본 뒤 검정·보정 방식을 바꾸지 않는다.

Ambiguity **LOW** — 요청이 부록 A 세 항목으로 특정돼 있다. Difficulty **Medium** — 새 실험 없음,
기존 산출물 재분석 스크립트 1개 + 문서 3개.

## 1. Objective

1. **참고문헌**: §10을 **확인된** 인용으로 채운다.
2. **영문 본문**: 초안 전체의 영문판을 별도 파일로 만든다.
3. **통계 검정**: 판정 표의 seed별 쌍 비교에 검정·신뢰구간·다중비교 보정을 붙인다.

## 2. Non-goals

- 새 실험·재실행 없음. 판정(verdict) 수정 없음 — 사전등록 규칙은 "≥4/5 seed 우위"이며 검정은 **보조 보고**다.
- 문항 단위(item-level) 검정 없음 — seed별 per-item 예측이 저장돼 있지 않다. 한계로 적는다.
- Paper 2 재개 아님.
- `pyproject.toml`·`uv.lock` 변경 없음. `scipy`는 이미 잠금 파일에 있는 전이 의존성으로 하네스 스크립트에서만 쓴다.

## 3. 통계 검정 — 계산 전 동결

**단위**: factor × 지표 쌍 **14개**(7 factor × 사전등록 지표 2개). 값은 각 factor의 `judgment.json` `per_seed`
— 판정이 실제로 본 수치 그대로다. 두 arm은 seed마다 초기 가중치를 공유하므로 **seed 단위 쌍 비교**다.

각 쌍에 대해 `d_s = hyper_s − control_s` (s = 1..5)로:

| 보고 항목 | 방법 |
|---|---|
| 평균 차이와 95% 신뢰구간 | `mean(d) ± t_{0.975,4} · sd(d)/√5` |
| **1차 검정** | **쌍체 t-검정, 양측** |
| 다중비교 보정 | **Holm–Bonferroni**, 14개 전체, α = 0.05 (임의 의존성에서 유효 — factor들은 독립이 아니다: `v1_baseline`과 `v2b_head`는 `hyper` 값을 공유한다) |
| 분포 가정 없는 확인 | **정확 부호뒤집기 순열 검정**(2⁵ = 32가지), 양측 |
| 효과 크기 | Cohen's `d_z = mean(d)/sd(d)` |

**미리 밝히는 한계**: n=5에서 정확 순열 검정의 양측 최소 p는 **2/32 = 0.0625**다. 즉 5/5 전승이어도
0.05를 **넘을 수 없다.** 이것은 n의 바닥이지 반증이 아니며, 순열 검정은 "t-검정의 정규성 가정에 기대지 않아도
방향이 일관된가"의 확인으로만 쓴다. 판정 근거로 쓰지 않는다.

**해석 규칙(동결)**:
- 14개 전부 Holm 보정 후 p < 0.05 → 논문에 "seed 쌍체 t-검정, Holm 보정 후 전부 유의"로 적는다.
- 일부가 실패 → **실패한 쌍을 이름으로 적고**, 해당 격차를 "5 seed에서 방향은 일관되나 보정 후 유의하지 않음"으로
  한정한다. 판정은 바꾸지 않는다. 검정 방식을 바꿔 재계산하지 않는다.
- `flat_text` 참조선·판정 밖 진단(3 seed)·Paper 2 수치에는 검정을 붙이지 않는다(판정 대상이 아니다).

## 4. 참고문헌 규칙

- **확인한 것만** 적는다. 확인 경로는 (a) 저장소의 서베이 PDF(Kim et al., KDD 2024) 참고문헌 목록,
  (b) 출판처·arXiv 페이지 웹 검색. 둘 다 안 되면 싣지 않는다.
- 본문에서 실제로 쓰는 주장에만 인용을 단다. 목록을 채우려고 인용하지 않는다.

## 5. Touched files

- 신규 `docs/charter-hnn-ontology-paper1-finishing.md` (이 문서)
- 신규 `.harness/scripts/paper1_statistics.py` → `outputs/hnn_ontology_paper1_stats/{stats.json,stats.md}`
- 신규 `tests/test_paper1_statistics.py` — 순열 p 바닥 0.0625, Holm 계산, 알려진 t-검정 값 고정
- 신규 `docs/hnn-ontology-paper1-draft-en.md` — 영문 본문
- 수정 `docs/hnn-ontology-paper1-draft.md` — §5 통계 절, §10 참고문헌, 부록 A
- 수정 `docs/hnn-ontology-paper1-consolidation.md` — 산출물 표

## 6. Checkpoints

`charter_frozen` → `stats_contract_tests` → `stats_run` → `references_verified` → `english_draft`
→ `verify_paper1_claims` + 전체 테스트 → 커밋·머지.

Rollback: 문서·스크립트 추가뿐이므로 브랜치 폐기로 충분하다.

## 7. 추가 범위 (2026-09-13) — 대한산업공학회지 원고 변환

사용자가 제출처를 **대한산업공학회지(국문 저널)**로 지정. 원고작성요령은 jkiie.org Instructions for Authors에서
확인했다(초기 검색에 걸린 `ksci.re.kr` 규정은 한국컴퓨터정보학회 것이라 쓰지 않음).

- 신규 `.harness/scripts/build_paper1_jkiie_docx.cjs` → `outputs/hnn_ontology_paper1_jkiie/paper1_jkiie_manuscript.docx`
- 규정 반영: 국문 본문 · 별도 표지 · 영문 초록 150단어 이내(스크립트가 초과 시 실패) · 영문 키워드 · 그림/표 제목과 내용 영문 ·
  그림 캡션 아래/표 제목 위 · 각주 없음 · 저자-연도 인용 · 영문 참고문헌 알파벳순 · 이중 간격 · 쪽 번호 하단 중앙
- 새 수치 없음. 모든 수치는 초안(§5)에서 가져온다.
- **저자 판단 사항(자동화하지 않음)**: 저자 정보, 코드 공개 여부, 생성형 AI 활용 고지 문구, 그림 사용의 AI 규정 적합성,
  HWP 템플릿 이식(로그인 필요).
