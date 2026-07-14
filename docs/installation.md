# Installation

AsteroScale currently installs directly from its GitHub repository.

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
