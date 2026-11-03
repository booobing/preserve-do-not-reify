"""The two arms as models (charter §4).

``hyper``   -> upstream ``StarE_Transformer`` on the qualifier-aware statement graph, unchanged.
``reified`` -> the same class with ``forward_base`` replaced so the encoder runs StarE's own
               triple-mode convolution over the role-edge graph (statement nodes appended to the
               entity table) while the decoder still reads the hyper arm's query tokens.

Why a class swap instead of a normal subclass: upstream ``StarE_Transformer.__init__`` calls
``super(self.__class__, self).__init__``, which recurses forever for any subclass. So the reified
arm is built as a plain upstream instance and then re-tagged with a class that adds the blank-node
state and the override. No upstream file changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .reify_stare import ReifiedGraph
from .upstream import Upstream


class ReifiedMixin:
    """Overrides for the reified arm; mixed in front of upstream `StarE_Transformer`."""

    num_real_entities: int
    blank_predicates: Tensor
    w_blank: nn.Parameter

    def init_reified(self, reified: ReifiedGraph, get_param) -> None:
        self.num_real_entities = int(reified.num_real_entities)
        self.register_buffer(  # type: ignore[attr-defined]
            "blank_predicates", torch.as_tensor(reified.blank_predicates, dtype=torch.long, device=self.device)
        )
        # Same initialiser upstream uses for its own (1, d) vectors (loop_rel, loop_ent).
        self.w_blank = get_param((1, self.emb_dim))  # type: ignore[attr-defined]

    def blank_init(self) -> Tensor:
        """x[b_i] = init_rel[p_i] + w_blank: the blank carries its predicate and one shared offset."""
        return self.init_rel.index_select(0, self.blank_predicates) + self.w_blank  # type: ignore[attr-defined]

    def forward_base(self, sub, rel, drop1, drop2, quals=None, embed_qualifiers: bool = False, return_mask: bool = False):
        """Upstream `StarEEncoder.forward_base`, triple branch, with two changes: the node table is
        [entities ; blanks] and the candidate table returned for scoring is the entity rows only."""
        if self.model_nm.endswith("transe"):  # type: ignore[attr-defined]
            raise NotImplementedError("reified arm supports the default (rotate) relation parametrisation only")
        r = self.init_rel  # type: ignore[attr-defined]
        x = torch.cat([self.init_embed, self.blank_init()], dim=0)  # type: ignore[attr-defined]
        x, r = self.conv1(x=x, edge_index=self.edge_index, edge_type=self.edge_type, rel_embed=r)  # type: ignore[attr-defined]
        x = drop1(x)
        if self.n_layer == 2:  # type: ignore[attr-defined]
            x, r = self.conv2(x=x, edge_index=self.edge_index, edge_type=self.edge_type, rel_embed=r)  # type: ignore[attr-defined]
            x = drop2(x)
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        candidates = x[: self.num_real_entities]
        if not embed_qualifiers:
            return sub_emb, rel_emb, candidates
        if quals is None:
            raise ValueError("embed_qualifiers=True needs the qualifier tensor")
        quals_ents = quals[:, 1::2].reshape(-1)
        quals_rels = quals[:, 0::2].reshape(-1)
        qual_obj_emb = torch.index_select(x, 0, quals_ents).view(sub_emb.shape[0], -1, sub_emb.shape[1])
        qual_rel_emb = torch.index_select(r, 0, quals_rels).view(rel_emb.shape[0], -1, rel_emb.shape[1])
        if not return_mask:
            return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, candidates
        mask = torch.zeros((sub.shape[0], quals.shape[1] + 2), dtype=torch.bool, device=self.device)  # type: ignore[attr-defined]
        mask[:, 2:] = quals == 0
        return sub_emb, rel_emb, qual_obj_emb, qual_rel_emb, candidates, mask


_REIFIED_CLASSES: dict[type, type] = {}


def reified_class(base: type) -> type:
    """`class ReifiedStarE_Transformer(ReifiedMixin, <upstream StarE_Transformer>)`, built once."""
    if base not in _REIFIED_CLASSES:
        _REIFIED_CLASSES[base] = type("ReifiedStarE_Transformer", (ReifiedMixin, base), {"model_name": "ReifiedStarE_Transformer"})
    return _REIFIED_CLASSES[base]


def build_hyper_model(up: Upstream, graph_repr: dict[str, np.ndarray], model_config: dict[str, Any]) -> nn.Module:
    return up.StarE_Transformer(graph_repr, model_config)


def build_reified_model(
    up: Upstream, graph_repr: dict[str, np.ndarray], model_config: dict[str, Any], reified: ReifiedGraph
) -> nn.Module:
    if model_config["STATEMENT_LEN"] != 3:
        raise ValueError("reified arm must be constructed with the triple-mode config (make_model_config)")
    if model_config["NUM_RELATIONS"] != reified.num_relations or model_config["NUM_ENTITIES"] != reified.num_real_entities:
        raise ValueError("model config disagrees with the reified graph's id spaces")
    model = up.StarE_Transformer(graph_repr, model_config)
    model.__class__ = reified_class(up.StarE_Transformer)
    model.init_reified(reified, up.get_param)
    return model


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def is_reified(model: nn.Module) -> bool:
    return isinstance(model, ReifiedMixin)
