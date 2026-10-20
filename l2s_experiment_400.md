# SmartLab L2S CPU experiment

Input directory: `/home/luopw24/HLS/datasets/SmartLab_Challenge_2020`.

Families are processed in order: brandimarte (10 instances), barnes (21),
dauzere (18). Each instance has 10 independent runs, for 490 runs in total.
The input machine IDs are zero-based; the runner explicitly passes
`--machine_index_base 0`. Job and machine counts come from each input file.

## Fixed experiment settings

| Setting | Value |
| --- | --- |
| Conda environment | l2s |
| Device | cpu |
| Model | checkpoints/incumbent_10x5.pth |
| Initial solution model | checkpoints/daniel/10x5+mix.pth |
| Initialization | DANIEL |
| Improvement iterations per run | 400 |
| Action samples per state per iteration | 50 |
| Initial schedules sampled | 100 |
| Initial schedules retained for search | 20 |
| Seed for run r | 50 + (r - 1) * 10000 |
| Memory type / size / k | individual / 600 / 15 |
| Reward type / tabu | yaoxin / False |

Other model and experiment settings retain the defaults in `parameters.py`.
`n_processes=50` denotes action samples, independently of the CPU worker count.

Checkpoint SHA-256 digests recorded during execution:

```text
incumbent_10x5.pth: 3cdb21bb7c85f78da5b6b1bc71342ddeee74bee15749fe1cfaaa2879ee6b3402
daniel/10x5+mix.pth: 326f763b6878a7c64b63c2a8ab5f06ce158b532d581a4d02dd39d68218274bcb
```

## CPU execution

The runner selects concurrency from CPU affinity, physical cores, the cgroup
CPU quota, and available memory, including the cgroup memory limit. It probes
resources again during scheduling (every 10 seconds by default), accounting
for memory already occupied by active children. Resource monitoring during this
experiment settled on 16 workers with one compute thread per worker after
measuring CPU utilization. The container CPU quota permits approximately 16
cores, and its memory limit is approximately 64 GiB.

Each run executes in an isolated child process. A run is written to the family
CSV only after all 400 iterations finish. The parent atomically updates the CSV.
Existing numeric cells are preserved on resume. Failed runs are retried once;
unfinished or failed cells remain pending rather than being fabricated.

## Outputs

The CSV layout follows `/home/luopw24/HLS/copt.csv` exactly:

`instance,run1,run2,run3,run4,run5,run6,run7,run8,run9,run10`

| Family | File | Instance labels |
| --- | --- | --- |
| brandimarte | /home/luopw24/HLS/l2s_brandimarte.csv | Mk01 through Mk10 |
| barnes | /home/luopw24/HLS/l2s_barnes.csv | Original labels such as mt10c1 and seti5xyz |
| dauzere | /home/luopw24/HLS/l2s_dauzere.csv | 01a through 18a |

Per-run logs are in `/home/luopw24/HLS/l2s_logs/`.
The main execution log is `/home/luopw24/HLS/l2s_smartlab_400.log`.
CSV values are final best makespans. The separate original L2S `result.csv`
and the reference `copt.csv` are not used as experiment results.

## Resume command

Run from `/home/luopw24/L2S-FJSP` after `conda activate l2s`:

```bash
python -u run_smartlab.py --families brandimarte barnes dauzere --runs 10 --runs-per-process 1 --workers 0 --threads-per-worker 1 --worker-memory-gb 2 --memory-reserve-gb 4 --base-seed 50 --model-type incumbent --model-j 10 --model-m 5 --n-processes 50 --init-method DANIEL --n-iterations 400 --max-retries 1
```

The default input, output, and log paths match the paths listed above.
The runner uses an output lock, streams per-run logs, and cleans up child
processes when stopped with Ctrl+C or SIGTERM.

The entry script activates `l2s` automatically:

```bash
bash /home/luopw24/L2S-FJSP/resume_smartlab.sh
```

To inspect pending work without starting any experiment or writing any CSV:

```bash
bash /home/luopw24/L2S-FJSP/resume_smartlab.sh --dry-run
```
