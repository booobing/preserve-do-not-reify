"""Paper 2 bridge: hypergraph embeddings -> frozen decoder via gated cross-attention (contract C10).

This cycle ships the contract-level modules only (memory, gated cross-attention, a toy decoder and the
hook-based injection path). Wiring a Hugging Face model is phase P5 (docs/charter-hnn-llm-bridge-v1.md).
"""
