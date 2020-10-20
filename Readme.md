# chmp - Support code for machine learning / data science experiments

- [`chmp.ds`](docs/ds.md): data science support
- [`chmp.parser`](docs/parser.md): helpers to write parsers using functional
  composition
- [`chmp.torch_util`](docs/torch_utils.md): helpers to write pytorch models

The individual modules are designed ot be easily copyable outside this
distribution. For example to use the parser combinators just copy the
`__init__.py` into the local project.


To install / run tests use:

```bash
# install the package
pip install chmp

# to run tests
pip install pytest
pytest --pyargs chmp
```
