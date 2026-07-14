"""Batch solving for many independent targets -- each target is a
completely separate solve() call with no shared state between targets, so
this just parallelizes that embarrassingly-parallel loop across processes.

Uses the 'spawn' multiprocessing start method rather than the Linux
default 'fork'. Slightly slower to start workers (each reimports the
module fresh rather than sharing memory via copy-on-write), but is the
safe choice if JAX ever enters the picture here later -- JAX's XLA runtime
holds internal threads, and forking a process with active threads is a
classic source of deadlocks. It's a reasonable default even without JAX.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def _init_worker():
    # Each worker process runs its own single-threaded solve() -- the
    # parallelism here is across processes, not within one. Without this,
    # numpy/scipy's BLAS backend (OpenBLAS/MKL) may also try to spawn
    # multiple threads *per process* for linear algebra, and N worker
    # processes each doing that oversubscribes the machine's cores,
    # eating into (or reversing) the speedup from parallelizing at all.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _solve_one(args):
    from .solver import Solver  # imported here, not at module level -- see
    # _init_worker: numpy/scipy need to see the thread-count env vars
    # *before* they're imported, and with the 'spawn' start method each
    # worker is a fresh interpreter, so this runs after _init_worker but
    # before Solver (and therefore numpy) is ever touched in this process.

    target_id, given, want, solver_kwargs, seed = args
    solver = Solver(seed=seed, **solver_kwargs)
    try:
        result = solver.solve(given, want)
        return target_id, result, None
    except Exception as exc:
        return target_id, None, f"{type(exc).__name__}: {exc}"


def solve_many(
    targets,
    want,
    priors=None,
    preset="standard",
    nlive=None,
    sample=None,
    bound=None,
    bootstrap=None,
    walks=None,
    update_interval=None,
    n_jobs=None,
    base_seed=0,
    show_progress=False,
):
    """Solve many independent targets in parallel, one process per target
    (up to n_jobs at a time).

    targets: dict {target_id: given_dict}, e.g.
        {"KIC 12345678": {"Teff": (5777, 50), "numax": (3090, 30), ...},
         "KIC 87654321": {...}}
    want: list of quantity names applied to every target, or a dict
        {target_id: want_list} for per-target requests.
    priors, preset, nlive, sample, bound: shared Solver settings used for every
        target -- a fresh Solver is built in each worker process, nothing
        is shared/reused across targets. Custom priors must be picklable
        (frozen scipy.stats distributions and the classes in priors.py
        are; anything JAX-jitted likely isn't).
    n_jobs: number of worker processes (default: os.cpu_count()).
    base_seed: each target gets its own reproducible seed (base_seed +
        index) rather than sharing one RNG stream across workers.
    show_progress: print a running "done so far" count.

    Returns {target_id: solve()_output}, in the same order as `targets`.
    A target whose solve() call raised gets {"_error": "..."} instead of
    aborting the whole batch -- check for that key rather than assuming
    every requested quantity is present for every target.
    """
    solver_kwargs = dict(
        priors=priors,
        preset=preset,
        nlive=nlive,
        sample=sample,
        bound=bound,
        bootstrap=bootstrap,
        walks=walks,
        update_interval=update_interval,
    )
    items = list(targets.items())
    want_for = want if isinstance(want, dict) else {tid: want for tid, _ in items}

    jobs = [
        (tid, given, want_for[tid], solver_kwargs, base_seed + i)
        for i, (tid, given) in enumerate(items)
    ]

    ctx = multiprocessing.get_context("spawn")
    results = {}
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_init_worker) as pool:
        futures = {pool.submit(_solve_one, job): job[0] for job in jobs}
        done = 0
        for future in as_completed(futures):
            target_id, result, error = future.result()
            results[target_id] = {"_error": error} if error else result
            done += 1
            if show_progress:
                print(f"\r{done}/{len(jobs)} targets done", end="", flush=True)
        if show_progress:
            print()

    # Return in the original target order rather than completion order.
    return {tid: results[tid] for tid, _ in items}
