"""Sequential run queue over the committed matrix (config.py).

Usage: CUDA_VISIBLE_DEVICES=<gpu> python src/phases/run_queue.py \
           QUEUE_INDEX N_QUEUES PHASES
  e.g. two shells:  CUDA_VISIBLE_DEVICES=0 python src/phases/run_queue.py 0 2 AC
                    CUDA_VISIBLE_DEVICES=1 python src/phases/run_queue.py 1 2 AC

Static assignment: matrix row i (within the selected phases) goes to
queue (i % N_QUEUES); the matrix is seed-major so paired arms finish
close together. Completed runs (metrics.json exists) are skipped, which
makes the queue resumable after interruption. A failed run is logged and
the queue continues - one OOM must not kill a multi-day queue; the exit
code reports whether any run failed.
"""
import os
import subprocess
import sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def main():
    qi, nq, phases = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    runs = [r for r in C.RUN_MATRIX if r["phase"] in phases]
    mine = [r for i, r in enumerate(runs) if i % nq == qi]
    os.makedirs(C.RUNS_ROOT, exist_ok=True)
    print(f"[queue{qi}] {len(mine)} runs assigned "
          f"(phases {phases}, {len(runs)} total)", flush=True)
    trainer = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "train.py")
    fails = []
    for k, r in enumerate(mine):
        if os.path.exists(os.path.join(C.RUNS_ROOT, r["run_id"],
                                       "metrics.json")):
            print(f"[queue{qi}] skip {r['run_id']} (done)", flush=True)
            continue
        log = os.path.join(C.RUNS_ROOT, r["run_id"] + ".log")
        print(f"[queue{qi}] ({k + 1}/{len(mine)}) start {r['run_id']}",
              flush=True)
        with open(log, "w") as lf:
            rc = subprocess.run(
                [sys.executable, trainer, r["arm"], str(r["seed"]),
                 r["protocol"], r["train_env"], r["backbone"], r["arch"]],
                stdout=lf, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            fails.append(r["run_id"])
            print(f"[queue{qi}] FAILED {r['run_id']} rc={rc} (see {log})",
                  flush=True)
        else:
            print(f"[queue{qi}] done {r['run_id']}", flush=True)
    print(f"[queue{qi}] queue complete, {len(fails)} failures: {fails}",
          flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
