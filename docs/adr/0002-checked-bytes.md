# 0002 — Checked uint64 bytes

Use integer bytes/counts internally with checked addition, multiplication and
subtraction. GiB inputs round down to bytes; packed int4 planes round up their
final half-byte. Trade-off: fractional storage is represented at its actual
allocation granularity, and enormous unrepresentable inputs fail instead of
receiving an approximate floating-point estimate. GiB and GB stay distinct.
