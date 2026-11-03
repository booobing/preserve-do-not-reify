"""HAHE on the StarE harness (docs/charter-hahe-same-harness-v1.md).

Runs the pinned upstream HAHE code (third_party/HAHE, MIT, ACL 2023) on exactly the rows the StarE
harness feeds its models: the same WD50K(100) statements after StarE's loader (qualifier pairs capped
at six, repeats kept), the same train+valid training graph, the same test queries (subject and
object of every test statement), the same candidate set (entities that ever appear as a subject or
object) and the same filtered ranking. HAHE's own preprocessed json files are not used.

Nothing in third_party/HAHE is edited; `data.py` converts rows and builds the evaluation index,
`runner.py` reproduces HAHE's training loop (run.py) with seeds, checkpoints and our evaluation.
"""
