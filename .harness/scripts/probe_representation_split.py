"""Same diagnostic as .harness/scripts/probe_bridge_memory.py, run for three representations.

hyper           - the n-ary fact kept as one hyperedge (the proposal)
reified_binary  - the same fact expanded into blank node + binary role edges (RDF/Wikidata's way)
flat_text       - no structure at all: the fact sentence through the text tower (an ordinary
                  embedding index, i.e. what a vector RAG store holds)

All three are trained the same way on the same world with the same budget and answer the same 118
QA test questions, so the only thing that differs is how a fact is represented.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run from anywhere

from src.ontology.baselines import ModelConfig, build_model  # noqa: E402
from src.ontology.contrastive import TrainConfig, WorldView, train_alignment  # noqa: E402
from src.ontology.llm_bridge.qa_data import build_qa_items  # noqa: E402
from src.ontology.llm_bridge.retriever import FactRetriever  # noqa: E402
from src.ontology.run_encoder_benchmark import enable_determinism, resolve_device  # noqa: E402
from src.ontology.synthetic import load_preset  # noqa: E402

K = 5
ARMS = ["hyper", "reified_binary", "flat_text"]
SEEDS = [1, 2, 3]
# Persist the rows. The v2 results doc has quoted this diagnostic since August with no artifact
# behind it, so nothing could check those numbers the way verify_paper1_claims.py checks the judged
# tables, and any figure using them would have had to hardcode values.
OUT = Path(__file__).resolve().parents[2] / "outputs" / "hnn_ontology_representation_split"

enable_determinism(True)
device = resolve_device("auto")
world = load_preset("v1")
view = WorldView.from_world(world)
items = [i for i in build_qa_items(world) if i.split == "test"]
graph = view.graph
rows = []

for arm_name in ARMS:
    for seed in SEEDS:
        cfg = ModelConfig()
        model = build_model(arm_name, view, cfg, seed=seed)
        train_alignment(model, view, world, TrainConfig(steps=1500, seed=seed), device=device)
        retriever = FactRetriever(model, view, device=device)
        model.eval()

        enc_hits = mem_hits = in_mem = ret_hits = 0
        with torch.no_grad():
            for item in items:
                fact, role = item.gold_edge_ids[0], item.masked_role
                q_emb, node_emb, node_index = model.completion_forward(view, [(fact, role)])
                query = q_emb[0]

                candidate_ids = view.candidates_by_type[view.role_type(fact, role)]
                cand = node_emb[[node_index[c] for c in candidate_ids]]
                enc_hits += int(candidate_ids[int((cand @ query).argmax())] == item.answer_node_id)

                retrieved = retriever.top_k(item.question, K).fact_ids
                ret_hits += int(fact in retrieved)
                injected = []
                for f in retrieved:
                    for node_id, _ in graph.edges[f].members:
                        if node_id not in injected:
                            injected.append(node_id)
                in_mem += int(item.answer_node_id in injected)
                pool = [n for n in injected if n in candidate_ids]
                if pool:
                    pool_emb = node_emb[[node_index[n] for n in pool]]
                    mem_hits += int(pool[int((pool_emb @ query).argmax())] == item.answer_node_id)

        n = len(items)
        row = {"arm": arm_name, "seed": seed, "encoder_acc@1": enc_hits / n,
               "retrieval_hit@5": ret_hits / n, "gold_in_memory": in_mem / n,
               "memory_acc@1": mem_hits / n}
        rows.append(row)
        print(json.dumps(row), flush=True)

KEYS = ("encoder_acc@1", "retrieval_hit@5", "gold_in_memory", "memory_acc@1")

print("\nMEANS")
means = {}
for arm_name in ARMS:
    sub = [r for r in rows if r["arm"] == arm_name]
    m = {k: sum(r[k] for r in sub) / len(sub) for k in KEYS}
    means[arm_name] = m
    print(f"{arm_name:<16} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "rows.jsonl").open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
(OUT / "summary.json").write_text(json.dumps(
    {"arms": ARMS, "seeds": SEEDS, "k": K, "n_items": len(items),
     "world_preset": "v1", "steps": 1500, "means": means, "rows": rows},
    ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'rows.jsonl'} and summary.json")
