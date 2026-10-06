# 0003 — Explicit vLLM-style budget

Subtract weights and separate non-torch, activation and graph allowances from
the utilization budget, then floor whole cache blocks. Defaults are explicitly
UNCALIBRATED; checkpoint sizes/engine logs should replace them. Trade-off: more
configuration is required, but hidden scratch multipliers do not masquerade as
engine knowledge. MLA weights require explicit sizes because cache dimensions
alone do not identify projection matrices.
