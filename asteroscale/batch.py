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
    """Limit numerical libraries to one thread in each worker process."""
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
    """Run one target inside a batch worker.

    Parameters
    ----------
    args : tuple
        ``(target_id, given, want, solver_kwargs, seed)`` job description.

    Returns
    -------
    tuple
        Target identifier, result or ``None``, and error string or ``None``.
    """
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
    bandpass="TESS",
    input_mode="propagate",
    relation_scatter=None,
    warn_validity=True,
    n_jobs=None,
    base_seed=0,
    show_progress=False,
):
    
    """Solve independent targets in parallel.

    Parameters
    ----------
    targets : dict
        Mapping from target identifiers to ``given`` dictionaries.
    want : sequence of str or dict
        Quantities requested for every target, or a mapping from target
        identifier to a target-specific request.
    priors : dict, optional
        Shared prior overrides. Custom distributions must be picklable.
    preset : {'fast', 'standard', 'precise'}, default='standard'
        Shared named sampling configuration.
    nlive : int, optional
        Shared number of live points.
    sample, bound : str, optional
        Shared Dynesty sampling and bounding methods.
    bootstrap, walks, update_interval : int, optional
        Additional shared Dynesty settings.
    bandpass : {'TESS', 'Kepler'}, default='TESS'
        Photometric response used for ``A_env``.
    input_mode : {'propagate', 'likelihood'}, default='propagate'
        Interpretation of uncertain fundamental inputs.
    relation_scatter : float or dict, optional
        Fractional intrinsic scatter for empirical relations. The default is
        zero for the ``fast`` and ``standard`` presets and the calibrated
        relation scatter for ``precise``.
    warn_validity : bool, default=True
        Warn about samples outside adopted calibration domains.
    n_jobs : int, optional
        Maximum worker processes. The default uses the available CPUs.
    base_seed : int, default=0
        Seed for the first target; subsequent targets add their index.
    show_progress : bool, default=False
        Print the number of completed targets.

    Returns
    -------
    dict
        Results in input order. Failed targets contain an ``"_error"`` key.
    """
    solver_kwargs = dict(priors=priors,
                         preset=preset,
                         nlive=nlive,
                         sample=sample,
                         bound=bound,
                         bootstrap=bootstrap,
                         walks=walks,
                         update_interval=update_interval,
                         bandpass=bandpass,
                         input_mode=input_mode,
                         relation_scatter=relation_scatter,
                         warn_validity=warn_validity,
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
