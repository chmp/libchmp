# `chmp.torch_utils`

## `chmp.torch_utils`

Helper to construct models with pytorch.


### `chmp.torch_utils.fixed`
`chmp.torch_utils.fixed(value)`

decorator to mark a parameter as not-optimized.


### `chmp.torch_utils.optimized`
`chmp.torch_utils.optimized(value)`

Decorator to mark a parameter as optimized.


### `chmp.torch_utils.optional_parameter`
`chmp.torch_utils.optional_parameter(arg, *, default=<class 'chmp.torch_utils.optimized'>)`

Make sure arg is a tensor and optionally a parameter.

Values wrapped with `fixed` are returned as a tensor, `values` wrapped
with `optimized``are returned as parameters. When arg is not one of
``fixed` or `optimized` it is wrapped with `default`.

Usage:

```python
class MyModule(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()

        # per default a will be optimized during training
        self.a = optional_parameter(a, default=optimized)

        # per default B will not be optimized during training
        self.b = optional_parameter(b, default=fixed)
```


### `chmp.torch_utils.t2n`
`chmp.torch_utils.t2n(obj=<undefined>, *, dtype=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), tensors=(<class 'torch.Tensor'>,))`

Torch to numpy.


### `chmp.torch_utils.n2t`
`chmp.torch_utils.n2t(obj=<undefined>, *, dtype=None, device=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), arrays=(<class 'numpy.ndarray'>, <class 'pandas.core.series.Series'>, <class 'pandas.core.frame.DataFrame'>))`

Numpy to torch.


### `chmp.torch_utils.t2t`
`chmp.torch_utils.t2t(func=<undefined>, *, dtype=None, returns=None, device=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), arrays=(<class 'numpy.ndarray'>, <class 'pandas.core.series.Series'>, <class 'pandas.core.frame.DataFrame'>), tensors=(<class 'torch.Tensor'>,))`

Equivalent  to `n2t(t2n(func)(*args, **kwargs)`


### `chmp.torch_utils.n2n`
`chmp.torch_utils.n2n(func=<undefined>, *, dtype=None, returns=None, device=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), arrays=(<class 'numpy.ndarray'>, <class 'pandas.core.series.Series'>, <class 'pandas.core.frame.DataFrame'>), tensors=(<class 'torch.Tensor'>,))`

Equivalent to `t2n(n2t(func)(*args, **kwargs)`


### `chmp.torch_utils.batched_n2n`
`chmp.torch_utils.batched_n2n(func, dtype=None, returns=None, device=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), arrays=(<class 'numpy.ndarray'>, <class 'pandas.core.series.Series'>, <class 'pandas.core.frame.DataFrame'>), tensors=(<class 'torch.Tensor'>,), batch_size=64)`

Wraper to call a torch function batch-wise with numpy args and results

This function behaves similar to [n2n](#n2n), but only supports function
arguments.

Usage:

```python
pred = batched_n2n(model, batch_size=128)(x)
```


### `chmp.torch_utils.optimizer_step`
`chmp.torch_utils.optimizer_step(optimizer, func=None, *args, **kwargs)`

Call the optimizer

This function can be used directly, as in:

```python
loss = optimizer_step(optimizer, evaluate_loss)
```

or as a decorator:

```python
@optimizer_step(optimizer)
def loss():
    return evaluate_loss()
```

The loss returned by evaluate_loss can be scalar loss, a tuple or a
dict. For a tuple, only the first element is used. For a dict, only the
"loss" item.


### `chmp.torch_utils.identity`
`chmp.torch_utils.identity(x)`

Return the argument unchanged


### `chmp.torch_utils.linear`
`chmp.torch_utils.linear(x, weights)`

A linear interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_utils.factorized_quadratic`
`chmp.torch_utils.factorized_quadratic(x, weights)`

A factorized quadratic interaction.

#### Parameters

* **x** (*any*):
  shape `(batch_size, in_features)`
* **weights** (*any*):
  shape `(n_factors, in_features, out_features)`


### `chmp.torch_utils.masked_softmax`
`chmp.torch_utils.masked_softmax(logits, mask, axis=-1, eps=1e-09)`

Compute a masked softmax


### `chmp.torch_utils.find_module`
`chmp.torch_utils.find_module(root, predicate)`

Find a (sub) module using a predicate.

#### Parameters

* **predicate** (*any*):
  a callable with arguments `(name, module)`.

#### Returns

the first module for which the predicate is true or raises
a `RuntimeError`.


### `chmp.torch_utils.DiagonalScaleShift`
`chmp.torch_utils.DiagonalScaleShift(shift=None, scale=None)`

Scale and shift the inputs along each dimension independently.


### `chmp.torch_utils.Identity`
`chmp.torch_utils.Identity()`

A module that does not modify its argument

Initializes internal Module state, shared by both nn.Module and ScriptModule.


### `chmp.torch_utils.Do`
`chmp.torch_utils.Do(func, **kwargs)`

Module that calls a function as a pure side-effect

The module can take additional keyword arguments. If the keyword arguments
are modules themselves or parameters they are found by .parameters().


### `chmp.torch_utils.Lambda`
`chmp.torch_utils.Lambda(func, **kwargs)`

Module that calls a function inline

The module can take additional keyword arguments. If the keyword arguments
are modules themselves or parameters they are found by .parameters().

For example, this module calls an LSTM and returns only the ouput, not the
state:

```python
mod = Lambda(
    lambda x, nn: nn(x)[0],
    nn=torch.nn.LSTM(5, 5),
)

mod(torch.randn(20, 10, 5))
```


### `chmp.torch_utils.LocationScale`
`chmp.torch_utils.LocationScale(activation=None, eps=1e-06)`

Split its input into a location / scale part

The scale part will be positive.


### `chmp.torch_utils.SplineBasis`
`chmp.torch_utils.SplineBasis(knots, order, eps=1e-06)`

Compute basis splines

Example:

```python
basis = SplineBasis(knots, order=3)
basis = torch.jit.script(basis)

x = np.linspace(0, 4, 100)
r = n2n(basis, dtype="float32")(x)

plt.plot(x, r)
```


### `chmp.torch_utils.SplineFunction`
`chmp.torch_utils.SplineFunction(knots, order)`

A function based on splines

Example:

```python
func = SplineFunction([0.0, 1.0, 2.0, 3.0, 4.0], order=3)
func = torch.jit.script(func)

optim = torch.optim.Adam(func.parameters(), lr=1e-1)

for _ in range(200):
    optim.zero_grad()
    loss = ((y - func(x)) ** 2.0).mean()
    loss.backward()
    optim.step()
```


### `chmp.torch_utils.make_mlp`
`chmp.torch_utils.make_mlp(in_features, out_features, *, hidden=(), hidden_activation=<class 'torch.nn.modules.activation.ReLU'>, activation=None, container=<class 'torch.nn.modules.container.Sequential'>)`

Build a feed-forward network


### `chmp.torch_utils.NumpyDataset`
`chmp.torch_utils.NumpyDataset(data, dtype=None, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), filter=None)`

A PyTorch datast composed out of structured numpy array.

#### Parameters

* **data** (*any*):
  the (structured) data. Nones are returned as is-is.
* **dtype** (*any*):
  if given a (structured) dtype to apply to the data
* **filter** (*any*):
  an optional boolean mask indicating which items are available
* **sequences** (*any*):
  see `chmp.ds.smap`
* **mappings** (*any*):
  see `chmp.ds.smap`


#### `chmp.torch_utils.NumpyDataset.filter`
`chmp.torch_utils.NumpyDataset.filter(func)`

Evaluate a filter on the full data set and set the filter of this dataset


### `chmp.torch_utils.kl_divergence__gamma__log_normal`
`chmp.torch_utils.kl_divergence__gamma__log_normal(p, q)`

Compute the kl divergence with a Gamma prior and LogNormal approximation.

Taken from C. Louizos, K. Ullrich, M. Welling "Bayesian Compression for Deep Learning"
https://arxiv.org/abs/1705.08665


### `chmp.torch_utils.ESGradient`
`chmp.torch_utils.ESGradient(parameters, *, n_samples=50, scale=0.1)`

Estimate the gradient of a function using Evolution Strategies

The gradient will be assigned to the `grad` property of the parameters.
This way any PyTorch optimizer can be used. As the tensors are manipulated
in-place, they must not require gradients. For modules or tensors call
`requires_grad_(False)` before using `ESGradient`. The return value will
be the mean and std of the loss.

Usage:

```python
grad_fn = ESGradient(model.parameters())
optimizer = torch.optim.Adam(model.parameters())

# ...
optimizer.zero_grad()
grad_fn(lambda: compute_loss(model))
optimizer.step()
```

#### Parameters

* **parameters** (*any*):
  the parameters as an iterable :param n_samples: the
  number of samples with which to estimate the gradient :param scale: the
  scale of the perturbation to use. Can be passed as a list with the same
  length as parameters to give different scales for each parameter.


### `chmp.torch_utils.update_moving_average`
`chmp.torch_utils.update_moving_average(alpha, average, value)`

Update iterables of tensors by an exponentially moving average.

If `average` and `value` are passed as module parameters, this function
can be used to make one module the moving average of the other module:

```python
target_value_function = copy.copy(value_function)
target_value_function.requires_grad_(False)

# ...

update_moving_average(
    0.9,
    target_value_function.parameters(),
    value_function.parameters(),
)
```

