"""StarE bridge (docs/charter-stare-input-representation-v1.md).

Runs the pinned upstream StarE code (third_party/StarE, MIT) on WD50K_100 with two input
representations of the same statements:

- ``hyper``   - upstream StarE unchanged: the message-passing graph keeps each statement as one
                qualifier-aware edge.
- ``reified`` - the graph the encoder sees is the chapter-5 ``reify()`` view (statement node +
                role edges); the Transformer decoder receives the same query tokens as ``hyper``.

Nothing in third_party/StarE is edited. Adaptation happens through ``shims`` (import-time
stand-ins for ``torch_scatter`` and ``wandb``) and a subclass built in ``model.py``.

Modules import torch lazily so the pure helpers stay cheap to test.
"""
