# Installation

AsteroScale currently installs directly from its GitHub repository. It
requires Python 3.10 or newer. The normal, uniform, and other general
probability distributions used by the solver come from
[Baldr](https://github.com/nielsenmb/Baldr). The pinned Baldr revision is
installed automatically from GitHub because Baldr is not yet on PyPI.

```bash
git clone https://github.com/nielsenmb/AsteroScale.git
cd AsteroScale
python -m pip install -e .
```

For development, tests and local documentation builds:

```bash
python -m pip install -e ".[test,docs]"
pytest
sphinx-build -W -b html docs docs/_build/html
```

The generated documentation will be available at
`docs/_build/html/index.html`.
