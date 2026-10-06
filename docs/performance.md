# Planner CPU performance

Measured CPU runtime/RSS of kvbench, not GPU inference throughput or calibration.

Host: macOS-26.6.2-arm64-arm-64bit-Mach-O, architecture arm64.
Binary: kvbench 1.0.0rc1, Release build.
CPU: Apple M2.

Command: `python3 bench/run.py --binary build-release/kvbench --runs 3`.

Toy geometry isolates planner overhead: one layer/head/dimension, explicit zero
weights/runtime allowances, 1 GiB pool; schedule prompts=32, outputs=8, 256
sequence slots and 4096 batched tokens. It is not a real checkpoint model.
Measured time includes process startup, JSON IO and formatting; stdout is discarded.

| Workload | Median wall time (ms) | Peak child RSS (MiB) | Runs |
| --- | ---: | ---: | ---: |
| 10,000-point independent sweep | 27.31 | 7.81 | 3 |
| 10,000-request schedule | 205.06 | 22.55 | 3 |

RSS is the maximum observed child-process RSS from the system time utility;
no cross-machine performance or GPU provisioning accuracy is implied.
