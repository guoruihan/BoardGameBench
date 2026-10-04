"""First four factorials: identical public input and atomic action space.

Shared head receives only the global state code and raw action identifiers.
Neither it nor the graph encoder computes pattern progress or simulated successors.
"""
import math
import numpy as np
import torch
from torch import nn

from boardbench.environments.harmonies import Harmonies, CELLS, CELL_INDEX, DELTAS
from boardbench.solvers.neural import PolicyNetwork
from boardbench.v12.neural import CompactSolver
from boardbench.v12.encoding import encode, CARD_VECTORS
from .registry import FIRST_ROUND, AVAILABLE, THROUGHPUT, NUMERIC

INPUT_SIZE = 1346
CARD_WIDTH = len(CARD_VECTORS[1])


def raw_action_ids():
    # kind, color(+null), object slot(+null), target cell(+null).
    rows = [(0, 6, i, 23) for i in range(3)]
    rows += [(1, color, 4, cell) for color in range(6) for cell in range(23)]
    rows += [(2, 6, i, 23) for i in range(3)]
    rows += [(3, 6, slot, cell) for slot in range(4) for cell in range(23)]
    rows += [(4, 6, 4 if i == 0 else i - 1, 23) for i in range(4)]
    rows += [(5, 6, 4, 23)]
    assert len(rows) == 241 and len(set(rows)) == 241
    return torch.tensor(rows, dtype=torch.long)


class DirectedBoardEncoder(nn.Module):
    def __init__(self, hidden, width=32, layers=2):
        super().__init__()
        assert CARD_WIDTH == 83 and INPUT_SIZE == 1346
        neighbors = [[CELL_INDEX.get((q + dq, r + dr), -1) for dq, dr in DELTAS] for q, r in CELLS]
        indices = torch.tensor(neighbors, dtype=torch.long)
        self.register_buffer('neighbor_indices', indices.clamp_min(0))
        self.register_buffer('neighbor_valid', indices >= 0)
        self.node_input = nn.Linear(17, width)
        self.cell_identity = nn.Embedding(23, width)
        self.card_input = nn.Linear(CARD_WIDTH + 34, width)
        self.message = nn.ParameterList([nn.Parameter(torch.empty(6, width, width)) for _ in range(layers)])
        self.node_update = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        for weight in self.message:
            nn.init.xavier_uniform_(weight)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Sequential(nn.Linear(23 * width + 7 * width + 374, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh())

    def forward(self, x):
        board = x[:, :391].reshape(-1, 23, 17)
        slots = x[:, 626:762].reshape(-1, 4, 34)
        market = x[:, 427:526].reshape(-1, 3, 33)
        market = torch.cat((market, torch.zeros_like(market[:, :, :1])), -1)
        cards = torch.cat((slots, market), 1)
        present = 1 - cards[:, :, 0]  # card-id one-hot zero means absent.
        patterns = x[:, 762:1343].reshape(-1, 7, CARD_WIDTH)
        card_state = torch.tanh(self.card_input(torch.cat((patterns, cards), -1))) * present[:, :, None]
        h = torch.tanh(self.node_input(board) + self.cell_identity.weight[None])
        valid = self.neighbor_valid[None, :, :, None]
        degree = self.neighbor_valid.sum(-1).clamp_min(1)[None, :, None]
        for weights, update in zip(self.message, self.node_update):
            adjacent = h[:, self.neighbor_indices] * valid
            messages = torch.einsum('bndh,dhk->bnk', adjacent, weights) / degree
            h = torch.tanh(update(h) + messages)
        scores = self.query(h) @ self.key(card_state).transpose(1, 2) / math.sqrt(h.shape[-1])
        attention = scores.masked_fill(~present.bool()[:, None, :], -1e9).softmax(-1)
        attention = attention * present[:, None, :]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-9)
        h = torch.tanh(h + attention @ self.value(card_state))
        remaining_raw = torch.cat((x[:, 391:762], x[:, -3:]), -1)
        return self.output(torch.cat((h.flatten(1), card_state.flatten(1), remaining_raw), -1))


class RawSharedHead(nn.Module):
    def __init__(self, hidden, width=64):
        super().__init__()
        self.register_buffer('identifiers', raw_action_ids())
        self.embeddings = nn.ModuleList([nn.Embedding(size, width) for size in (6, 7, 5, 24)])
        self.global_query = nn.Linear(hidden, width)
        self.score = nn.Linear(width, 1)
        nn.init.normal_(self.score.weight, std=.01)
        nn.init.zeros_(self.score.bias)

    def forward(self, state):
        candidates = sum(embedding(self.identifiers[:, i]) for i, embedding in enumerate(self.embeddings))
        return self.score(torch.tanh(self.global_query(state)[:, None, :] + candidates[None])).squeeze(-1)


class ResearchNetwork(nn.Module):
    def __init__(self, state, head, hidden, graph_width, graph_layers, action_width):
        super().__init__()
        self.body = (DirectedBoardEncoder(hidden, graph_width, graph_layers) if state == 'graph' else
                     nn.Sequential(nn.Linear(INPUT_SIZE, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()))
        self.actor = RawSharedHead(hidden, action_width) if head == 'shared' else nn.Linear(hidden, 241)
        self.critic = nn.Linear(hidden, 1)
        if head == 'fixed':
            nn.init.normal_(self.actor.weight, std=.01)
            nn.init.zeros_(self.actor.bias)

    def forward(self, x, mask):
        if x.shape[-1] != INPUT_SIZE or mask.shape[-1] != 241 or not mask.any(-1).all():
            raise ValueError('research network shape or empty legal mask')
        state = self.body(x.to(dtype=next(self.parameters()).dtype))
        return self.actor(state).masked_fill(~mask, -1e9), self.critic(state).squeeze(-1)


class NumericPolicyNetwork(PolicyNetwork):
    """Same parameter names and initialization; cast public FP32 input at entry."""
    def forward(self, x, mask):
        return super().forward(x.to(dtype=next(self.parameters()).dtype), mask)


class ResearchSolver(CompactSolver):
    def __init__(self, seed, experiment_id='E00', state='flat', head='fixed', hidden=128,
                 graph_width=32, graph_layers=2, action_width=64, device='cpu', numeric_dtype='float32'):
        parent = NUMERIC.get(experiment_id, experiment_id)
        base_id = THROUGHPUT.get(parent, parent)
        if experiment_id not in AVAILABLE or (state, head) != FIRST_ROUND[base_id][1:3]:
            raise ValueError('experiment and architecture mismatch or planned-only method')
        if numeric_dtype != ('float64' if experiment_id in NUMERIC else 'float32'):
            raise ValueError('numeric dtype differs from registered experiment')
        self.device = str(device)
        self.params = dict(action_encoding='slots240_v1', observation_encoding='cards_v1',
                           allow_stop=True, time_conditioning=False)
        self.metadata = dict(format_version=121, task_id='harmonies', rules_version=Harmonies.rules_version,
                             network='harmonies_factorial_v1', input_size=INPUT_SIZE, action_count=241,
                             experiment_id=experiment_id, state=state, head=head, hidden=hidden,
                             graph_width=graph_width, graph_layers=graph_layers, action_width=action_width,
                             **self.params)
        if experiment_id in NUMERIC:
            self.metadata['numeric_dtype'] = numeric_dtype
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed % (2**63))
            policy_class = NumericPolicyNetwork if experiment_id in NUMERIC else PolicyNetwork
            self.model = (policy_class(INPUT_SIZE, 241, hidden) if base_id == 'E00' else
                          ResearchNetwork(state, head, hidden, graph_width, graph_layers, action_width))
        self.model.to(device=device, dtype=torch.float64 if experiment_id in NUMERIC else torch.float32).eval()
        self.training = {}
        if len(encode(Harmonies().observe())['features']) != INPUT_SIZE:
            raise ValueError('public observation feature layout changed')
