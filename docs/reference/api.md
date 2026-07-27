# API reference

The API documentation below is generated from the NumPy-style docstrings in
the AsteroScale source code.

## Main interface

```{eval-rst}
.. autofunction:: asteroscale.solve

.. autofunction:: asteroscale.solve_many

.. autofunction:: asteroscale.summarize

.. autofunction:: asteroscale.plot_posterior
```

## Solver

```{eval-rst}
.. autoclass:: asteroscale.Solver
   :members:
   :show-inheritance:
```

## Scaling relations

```{eval-rst}
.. automodule:: asteroscale.relations
   :members:
   :member-order: bysource
```

## Sampling settings

```{eval-rst}
.. automodule:: asteroscale.sampling
   :members:
   :member-order: bysource
```

## Calibration settings

```{eval-rst}
.. automodule:: asteroscale.calibration
   :members:
   :member-order: bysource
```

## Calibration validity

```{eval-rst}
.. automodule:: asteroscale.validity
   :members:
   :member-order: bysource
```

## Offline population-prior training

These interfaces create portable GMM files and are not called by the solver.
See {doc}`../development/population-prior-training` before using them.

```{eval-rst}
.. automodule:: asteroscale.training
   :members:
   :member-order: bysource
```
