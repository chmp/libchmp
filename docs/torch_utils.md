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

```
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

```
pred = batched_n2n(model, batch_size=128)(x)
```


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


### `chmp.torch_utils.Do`
`chmp.torch_utils.Do(func, **kwargs)`

Call a function as a pure side-effect.


### `chmp.torch_utils.LookupFunction`
`chmp.torch_utils.LookupFunction(input_min, input_max, forward_values, backward_values)`

Helper to define a lookup function incl. its gradient.

Usage:

```
import scipy.special

x = np.linspace(0, 10, 100).astype('float32')
iv0 = scipy.special.iv(0, x).astype('float32')
iv1 = scipy.special.iv(1, x).astype('float32')

iv = LookupFunction(x.min(), x.max(), iv0, iv1)

a = torch.linspace(0, 20, 200, requires_grad=True)
g, = torch.autograd.grad(iv(a), a, torch.ones_like(a))
```


### `chmp.torch_utils.kl_divergence__gamma__log_normal`
`chmp.torch_utils.kl_divergence__gamma__log_normal(p, q)`

Compute the kl divergence with a Gamma prior and LogNormal approximation.

Taken from C. Louizos, K. Ullrich, M. Welling "Bayesian Compression for Deep Learning"
https://arxiv.org/abs/1705.08665


### `chmp.torch_utils.ESGradient`
`chmp.torch_utils.ESGradient(parameters, *, n_samples=50, scale=0.5)`

Estimate the gradient of a function using Evolution Strategies

The gradient will be assigned to the `grad` property of the parameters.
This way any PyTorch optimizer can be used. As the tensors are manipulated
in-place, they must not require gradients. For modules or tensors call
`requires_grad_(False)` before using `ESGradient`. The return value will
be the mean and std of the loss.

Usage:

```
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

```
target_value_function = copy.copy(value_function)
target_value_function.requires_grad_(False)

# ...

update_moving_average(
    0.9,
    target_value_function.parameters(),
    value_function.parameters(),
)
```

