# Preserve, Do Not Reify

Code, run artifacts, plans and checks for the manuscript
**"Preserve, Do Not Reify: Controlled Evidence for Keeping N-ary Facts Whole in Knowledge Graph Completion"**
([`papers/preserve_vs_reify/preserve_vs_reify.pdf`](papers/preserve_vs_reify/preserve_vs_reify.pdf)).

The paper holds a link-prediction model fixed and changes only how an n-ary fact enters it: whole (a hyperedge,
or StarE's qualifier-carrying edge) or as its standard reification (a statement node joined to each participant
by a binary edge). Every judged comparison is decided by one rule written down before its results were seen:
an arm wins a seed if it is strictly better on it, and wins the comparison only if it wins **both** judged
metrics on at least 4 of 5 seeds. Section 4.4 of the manuscript states how far the timing of each plan can be
verified.

## Results at a glance

| Study | Comparison | Result (mean of seeds 1–5) | Seeds won |
|---|---|---|---|
| 1 | One role-aware hypergraph encoder, hyperedge (`preserve`) vs. reified graph, seven conditions (synthetic world and WD50K(100)) | `preserve` ahead in every condition, e.g. baseline recall@5 / acc@1 0.863 / 0.502 vs. 0.670 / 0.424 | 5/5 on both metrics, every condition |
| 2 | Unmodified StarE code, native graph vs. reified graph, WD50K(100), StarE protocol, 401 epochs | test MRR 0.6553 vs. 0.6485; Hits@10 0.7766 vs. 0.7708 | 5/5, 5/5 |
| 3 | Our encoder inside the StarE pipeline: `preserve` vs. `reify` vs. `prune` (qualifiers removed from the encoder) | test MRR 0.6648 vs. 0.6561 vs. 0.6519 | 5/5 against each control |
| Position | HAHE's public code on the identical pipeline vs. our encoder (`preserve`) | test MRR 0.6784 vs. 0.6648; Hits@10 0.7947 vs. 0.7777 | HAHE 5/5 |

The last row compares models, not representations: our encoder is a vehicle for a controlled comparison, not a
competitive model.

## Repository layout

Paths are the ones of the research repository the runs came from, so every script runs unchanged.

| Path | Contents |
|---|---|
| `src/ontology/` | Study 1 encoder, synthetic world, WD50K loader, training and evaluation (`run_encoder_benchmark_v2.py`) |
| `src/ontology/stare_bridge/` | Studies 2–3: runs the pinned StarE code, the reified graph and our encoder inside StarE's pipeline |
| `src/ontology/hahe_bridge/` | HAHE comparison: feeds HAHE's unmodified code the rows, queries, candidates and scoring of Studies 2–3 |
| `third_party/*.lock.json` | Pinned commits and file hashes of StarE and HAHE (their code is fetched, not redistributed) |
| `.harness/scripts/` | Fetching, the Windows run driver, the judge, the claim and number checks, statistics, figures, manuscript build |
| `tests/` | Contract tests (65) for the encoder, the StarE and HAHE bridges and the judge |
| `outputs/` | Run artifacts: per-epoch metrics, per-run summaries and configurations, judgment files, statistics, figures |
| `docs/` | Plans ("charters") and result reports for each study, in Korean |
| `papers/preserve_vs_reify/` | Manuscript source, bibliography and built PDF |

Not included: model checkpoints and logs, the StarE and HAHE code (fetched at pinned commits), WD50K(100)
(downloaded from the StarE release).

## Environment

Python 3.13 and [uv](https://docs.astral.sh/uv/). The lock pins every package; on Windows it installs the
PyTorch 2.10 CUDA 13.0 wheel, elsewhere PyTorch 2.10 from PyPI.

```bash
uv sync --frozen
```

Always run with `uv run --frozen` so the environment stays exactly the locked one.

Fetch the data and the pinned upstream code (network access needed once). WD50K(100) must come first, because
the StarE check compares StarE's copy with it:

```bash
uv run --frozen python -c "from src.ontology.real_data import fetch_wd50k; fetch_wd50k()"
uv run --frozen python .harness/scripts/fetch_stare.py
uv run --frozen python .harness/scripts/fetch_hahe.py
```

## Check the paper without training (a few minutes)

```bash
uv run --frozen pytest -q

# Studies 2 and 3 and the HAHE comparison: recompute each verdict from the stored per-run summaries
uv run --frozen python .harness/scripts/judge_stare_input_representation.py
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other native_reified
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other native_prune
uv run --frozen python .harness/scripts/judge_stare_input_representation.py --root outputs/stare_native_encoder_v1/judged --base native_hyper --other hahe --other-root outputs/hahe_same_harness_v1/judged --other-epochs 300

# Study 1: every headline number against its artifacts, and the seed-paired tests
uv run --frozen python .harness/scripts/verify_paper1_claims.py
uv run --frozen python .harness/scripts/paper1_statistics.py

# every decimal number in the manuscript must appear in a source document
uv run --frozen python .harness/scripts/check_paper1_manuscript_numbers.py
```

The judge and the statistics script rewrite their output files under `outputs/`, so `git diff` afterwards shows
whether anything changed.

### What we got (2026-11-03, Windows 11, RTX 3060 Ti 8 GB)

| Check | Result |
|---|---|
| Copied files vs. the research repository | 366 of 366 identical (before the line-ending fix below) |
| `uv sync --frozen` | 89 packages, every version the same as in the research environment |
| Fetch checks | StarE `b294b9e2` and HAHE `e35e0a7b` verified |
| `pytest -q` | 65 passed, none skipped |
| Verdicts | Study 2 `hyper_wins` (StarE port 0.6553 vs. published 0.654); Study 3 `native_hyper_wins` twice; HAHE comparison `hahe_wins` |
| Recomputed files | J1, J2, J3 and the statistics files byte-identical; Study 2's `judgment.json` differs only in its `root` path and an `arms` field that the current judge adds |
| `verify_paper1_claims.py` | 44 claims across 7 conditions match |
| `paper1_statistics.py` | 14/14 pairs significant after Holm correction, as in the paper |
| `check_paper1_manuscript_numbers.py` | 262 of 262 decimal numbers found in the sources |
| Study 1 retrained (baseline, seed 1, both arms) | Loss history and all 42 summary values bit-identical to the stored run; only wall-clock fields differ (about 4 minutes) |

## Reproduce the runs

### Study 1

Runs in PyTorch's deterministic mode (the code sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Condition C uses a frozen
sentence encoder at a pinned revision and runs offline; download it once:

```bash
uv run --frozen python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42')"
```

```bash
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v1_baseline      --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2a_inductive    --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2b_head         --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2c_text         --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2d_real         --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2_combined      --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
uv run --frozen python -m src.ontology.run_encoder_benchmark_v2 --factor v2e_depth_parity --seeds 1 2 3 4 5 --out outputs/hnn_ontology_encoder_v2 --device auto
```

Running these into `outputs/hnn_ontology_encoder_v2` overwrites the stored artifacts; use another `--out` to
compare. Factor names map to the paper as: baseline, A inductive, B head parity, C frozen text, D real data,
A+B+C (report-only), depth + parity.

### Studies 2 and 3 and the HAHE comparison

Each run trains on train + validation statements and is scored on the full WD50K(100) test set (10,594 queries).
One run takes 9–14 hours on the GPU above. Runs checkpoint every five epochs and resume with `--resume`. The
150 ms GPU idle after each optimiser step was added to stop hardware resets on our machine; it does not change
the computation.

Any OS, one run at a time (`<arm>`, `<k>` and `<study>` as below):

```bash
uv run --frozen python -m src.ontology.stare_bridge.runner --arm <arm> --seed <k> --epochs 401 --eval-every 10 --chunk 5 --device cuda --use-test --resume --step-sleep-ms 150 --out outputs/<study>/judged/<arm>/seed_<k>
uv run --frozen python -m src.ontology.hahe_bridge.runner  --arm hahe  --seed <k> --epochs 300 --eval-every 10 --chunk 5 --device cuda --use-test --resume --step-sleep-ms 150 --out outputs/hahe_same_harness_v1/judged/hahe/seed_<k>
```

| Study | `<study>` | Arms | Epochs |
|---|---|---|---|
| 2 | `stare_input_representation_v1` | `hyper` (StarE's native graph), `reified` | 401 |
| 3 | `stare_native_encoder_v1` | `native_hyper` (`preserve`), `native_reified` (`reify`), `native_prune` (`prune`) | 401 |
| HAHE | `hahe_same_harness_v1` | `hahe` | 300 |

On Windows, the driver runs all seeds of a study in sequence, skips finished runs and resumes partial ones after
a reset:

```powershell
powershell -File .harness/scripts/run_stare_judged.ps1 -Detach
powershell -File .harness/scripts/run_stare_judged.ps1 -Detach -Arms native_hyper,native_reified,native_prune -OutRoot outputs\stare_native_encoder_v1\judged
powershell -File .harness/scripts/run_stare_judged.ps1 -Detach -Module src.ontology.hahe_bridge.runner -Arms hahe -Epochs 300 -OutRoot outputs\hahe_same_harness_v1\judged
```

The driver waits until the GPU power limit is at most 160 W (`-MaxPowerW`) and stops if the SM clock exceeds
1,450 MHz (`-MaxClockMHz`); these guards are for our machine and can be turned off with `-SkipGpuGuard`.

Study 2's 76-epoch pilot (one seed, validation set) is stored in `outputs/stare_input_representation_v1/{hyper,reified}/seed_0`;
its budget rule is in `docs/charter-stare-input-representation-v1.md` §5-b.

### Figures and manuscript

```bash
uv run --frozen python .harness/scripts/probe_representation_split.py   # Study 1 diagnostic behind Figure 4
uv run --frozen python .harness/scripts/make_paper1_figures.py          # Figures 1, 2 and 4
uv run --frozen python .harness/scripts/make_paper1_curve_figure.py     # Figure 3 (learning curves)
```

`.harness/scripts/build_paper1_manuscript.sh` expects a TinyTeX installation under `.harness/tools/`; with any
other TeX distribution run `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` on `papers/preserve_vs_reify/preserve_vs_reify.tex`.

## Plans and reports

Each study has a written plan (arms, metrics, budget, seeds, judging rule, predictions) and a report written from
the artifacts. They are in Korean.

| Study | Plan | Report |
|---|---|---|
| 1 | `docs/charter-hnn-ontology-program-v1.md` (baseline), `docs/charter-hnn-ontology-encoder-v2.md` (conditions A–D, combined), `docs/charter-hnn-ontology-encoder-v2e-depth-parity.md` (depth); contracts in `docs/hnn-ontology-contracts-v1.md`; tests in `docs/charter-hnn-ontology-paper1-finishing.md` §3 | `docs/hnn-ontology-encoder-v2-results.md`, `docs/hnn-ontology-paper1-draft-en.md` |
| 2 | `docs/charter-stare-input-representation-v1.md` | `docs/stare-input-representation-v1-results.md` |
| 3 | `docs/charter-stare-native-encoder-v1.md` | `docs/stare-native-encoder-v1-results.md` |
| HAHE | `docs/charter-hahe-same-harness-v1.md` | `docs/hahe-same-harness-v1-results.md` |
| Literature | — | `docs/related-work-hkg-comparison.md` |

The plans were written in a separate research repository; this repository is a copy, so its own history starts
after the runs.

## Differences from the research repository

- Code, tests, scripts, documents and artifacts are identical copies (up to line endings), with one exception:
  `sha256_of` in `src/ontology/stare_bridge/upstream.py` now normalises CRLF to LF, and the two lock files carry
  the matching hashes. The original check only passed on a Windows checkout with `core.autocrlf=true`.
  Data loading and training are untouched.
- `pyproject.toml` keeps only the dependencies this paper uses. `uv.lock` was relocked with the original
  cutoffs, so every remaining package has the same version.

## License

MIT for the code in this repository (see `LICENSE`). StarE (Galkin et al., 2020) and HAHE (Luo et al., 2023)
are both MIT-licensed. They are fetched from their repositories at pinned commits and not redistributed here.
WD50K(100) is part of the StarE release.
