## `chmp.ds`

Helpers for data science.

Distributed as part of `https://github.com/chmp/misc-exp` under the MIT
license, (c) 2017 Christopher Prohm.


### `chmp.ds.define`
`chmp.ds.define(func)`

Execute a function and return its result.

The idea is to use function scope to prevent pollution of global scope in
notebooks.

Usage:

```python
@define
def foo():
    return 42

assert foo == 42
```


### `chmp.ds.cached`
`chmp.ds.cached(path, validate=False)`

Similar to `define`, but cache to a file.

#### Parameters

* **path** (*any*):
  the path of the cache file to use
* **validate** (*any*):
  if True, always execute the function. The loaded result will be
  passed to the function, when the cache exists. In that case the
  function should return the value to use. If the returned value is
  not identical to the loaded value, the cache is updated with the
  new value.

Usage:

```python
@cached('./cache/result')
def dataset():
    ...
    return result
```

or:

```python
@cached('./cache/result', validate=True)
def model(result=None):
    if result is not None:
        # running to validate ...

    return result
```


### `chmp.ds.Object`
`chmp.ds.Object(*args, **kwargs)`

Dictionary-like namespace object.


### `chmp.ds.daterange`
`chmp.ds.daterange(start, end, step=None)`

A range of dates.


### `chmp.ds.undefined`
`chmp.ds.undefined()`

A sentinel to mark undefined argument values.


### `chmp.ds.first`
`chmp.ds.first(iterable, default=<undefined>)`

Return the first item of an iterable


### `chmp.ds.last`
`chmp.ds.last(iterable, default=<undefined>)`

Return the last item of an iterable


### `chmp.ds.nth`
`chmp.ds.nth(iterable, n, default=<undefined>)`

Return the nth value in an iterable.


### `chmp.ds.collect`
`chmp.ds.collect(iterable)`

Collect an iterable of `key, value` pairs into a dict of lists.


### `chmp.ds.cell`
`chmp.ds.cell(name=None)`

No-op context manager to allow indentation of code


### `chmp.ds.colorize`
`chmp.ds.colorize(items, *, skip=0, cmap=None, **kwargs)`

Given an iterable, yield `(color, item)` pairs.

#### Parameters

* **cmap** (*any*):
  if None the color cycle is used, otherwise it is interpreted as a
  colormap to color the individual items.
  
  Note: `items` is fully instantiated during the iteration. For any
  `list` or `tuple` item only its first element is used for
  colomapping.
  
  This procedure allows for example to colormap a pandas Dataframe
  grouped on a number column:
  
  ```python
  for c, (_, g) in colorize(df.groupby("g"), cmap="viridis"):
      ...
  ```
* **skip** (*any*):
  the nubmer of items to skip in the color cycle (only used for cmap is None)


### `chmp.ds.get_color_cycle`
`chmp.ds.get_color_cycle(n=None)`

Return the matplotlib color cycle.

#### Parameters

* **n** (*Optional[int]*):
  if given, return a list with exactly n elements formed by repeating
  the color cycle as necessary.

Usage:

```python
blue, green, red = get_color_cycle(3)
```


### `chmp.ds.mpl_axis`
`chmp.ds.mpl_axis(*args, **kwds)`

Set various style related options of MPL.

#### Parameters

* **xformatter** (*any*):
  if given a formatter for the major x ticks. Should have the
  signature `(x_value, pos) -> label`.
* **yformatter** (*any*):
  See `xformatter`.
* **invert** (*any*):
  if given invert the different axes. Can be x, y, or xy.


### `chmp.ds.errorband`
`chmp.ds.errorband(data, *, y, yerr, x=None, **kwargs)`

Plot erros as a band around a line

Usage:

```python
df.pipe(errorband, y="mean", yerr="std")
```


### `chmp.ds.diagonal`
`chmp.ds.diagonal(df=None, *, x, y, type='scatter', ax=None, **kwargs)`

Create a diagonal plot

#### Parameters

* **type** (*any*):
  the type to plot. Possible type "scatter", "hexbin".


### `chmp.ds.edges`
`chmp.ds.edges(x)`

Create edges for use with pcolor.

Usage:

```python
assert x.size == v.shape[1]
assert y.size == v.shape[0]
pcolor(edges(x), edges(y), v)
```


### `chmp.ds.center`
`chmp.ds.center(u)`

Compute the center between edges.


### `chmp.ds.axtext`
`chmp.ds.axtext(*args, **kwargs)`

Add a text in axes coordinates (similar `figtext`).

Usage:

```python
axtext(0, 0, 'text')
```


### `chmp.ds.index_query`
`chmp.ds.index_query(obj, expression, scalar=False)`

Execute a query expression on the index and return matching rows.

#### Parameters

* **scalar** (*any*):
  if True, return only the first item. Setting `scalar=True`
  raises an error if the resulting object has have more than one
  entry.


### `chmp.ds.query`
`chmp.ds.query(_df, *args, **kwargs)`

Filter a dataframe.

Usage:

```python
df.pipe(query, lambda df: df["col"] == "foo")
df.pipe(query, col="foo", bar="baz")
```


### `chmp.ds.fix_categories`
`chmp.ds.fix_categories(s, categories=None, other_category=None, inplace=False, groups=None, ordered=False)`

Fix the categories of a categorical series.

#### Parameters

* **s** (*pd.Series*):
  the series to normalize
* **categories** (*Optional[Iterable[Any]]*):
  the categories to keep. The result will have categories in the
  iteration order of this parameter. If not given but `groups` is
  passed, the keys of `groups` will be used, otherwise the existing
  categories of `s` will be used.
* **other_category** (*Optional[Any]*):
  all categories to be removed wil be mapped to this value, unless they
  are specified specified by the `groups` parameter. If given and not
  included in the categories, it is appended to the given categories.
  For a custom order, ensure it is included in `categories`.
* **inplace** (*bool*):
  if True the series will be modified in place.
* **groups** (*Optional[Mapping[Any,Iterable[Any]]]*):
  if given, specifies which categories to replace by which in the form
  of `{replacement: list_of_categories_to_replace}`.
* **ordered** (*bool*):
  if True the resulting series will have ordered categories.


### `chmp.ds.find_high_frequency_categories`
`chmp.ds.find_high_frequency_categories(s, min_frequency=0.02, n_max=None)`

Find categories with high frequency.

#### Parameters

* **min_frequency** (*float*):
  the minimum frequency to keep
* **n_max** (*Optional[int]*):
  if given keep at most `n_max` categories. If more are present after
  filtering for minimum frequency, keep the highest `n_max` frequency
  columns.


### `chmp.ds.as_frame`
`chmp.ds.as_frame(*args, **kwargs)`

Build a dataframe from kwargs or positional args.

Note, functions can be passed as kwargs. They will be evaluated with the
current dataframe and their result assigned to the named column. For
example:

```python
as_frame(
    x=np.random.uniform(-3, 3, 1_000),
    y=lambda df: np.random.normal(df["x"], 0.5),
)
```


### `chmp.ds.setdefaultattr`
`chmp.ds.setdefaultattr(obj, name, value)`

`dict.setdefault` for attributes


### `chmp.ds.transform_args`
`chmp.ds.transform_args(func, args, kwargs, transform, **transform_args)`

Transform the arguments of the function.

The arguments are normalized into a dictionary before being passed to the
transform function. The return value is a tuple of `args, kwargs` ready to
be passed to `func`.


### `chmp.ds.szip`
`chmp.ds.szip(iterable_of_objects, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,), combine=<class 'list'>)`

Zip but for deeply nested objects.
For a list of nested set of objects return a nested set of list.


### `chmp.ds.smap`
`chmp.ds.smap(func, arg, *args, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,))`

A structured version of map.
The structure is taken from the first arguments.


### `chmp.ds.copy_structure`
`chmp.ds.copy_structure(template, obj, sequences=(<class 'tuple'>,), mappings=(<class 'dict'>,))`

Arrange `obj` into the structure of `template`.

#### Parameters

* **template** (*any*):
  the object of which to copy the structure
* **obj** (*any*):
  the object which to arrange into the structure. If it is
  already structured, the template structure and its structure
  must be the same or a value error is raised


### `chmp.ds.json_numpy_default`
`chmp.ds.json_numpy_default(obj)`

A default implementation for `json.dump` that deals with numpy datatypes.


### `chmp.ds.patch`
`chmp.ds.patch(*args, **kwds)`

Monkey patch an object.

Usage:

```python
import sys
with patch(sys, argv=["dummy.py", "foo", "bar]):
    args = parser.parse_args()
```

After the block, the original values will be restored. Note: If an
attribute was previously not defined, it will be deleted after the block.


### `chmp.ds.groupby`
`chmp.ds.groupby(a)`

Group a numpy array.

Usage:

```python
for idx in grouped(arr):
    ...
```


### `chmp.ds.timed`
`chmp.ds.timed(tag=None, level=20)`

Time a codeblock and log the result.

Usage:

```python
with timed():
    long_running_operation()
```

The returned result can be used to estimate the remaining runtime:

```python
with timed() as timer:
    timer(0.5)
```

#### Parameters

* **tag** (*any*):
  an object used to identify the timed code block. It is printed with
  the time taken.


### `chmp.ds.status`
`chmp.ds.status(**items)`

Print (and update) a status line

Usage:

```python
status(epoch=epoch, loss=("{:.2f}", loss), complex=callable)
```

The print call is debounced to only execute only twice per second and to
take up at most 120 characters. To change these settings use the config
function:

```python
status.config(width=80, interval=2)
```


### `chmp.ds.find_categorical_columns`
`chmp.ds.find_categorical_columns(df)`

Find all categorical columns in the given dataframe.


### `chmp.ds.magic_open`
`chmp.ds.magic_open(p, mode, *, compression=None, atomic=False)`

Open compressed and  uncompressed files with a unified interface.

#### Parameters

* **compression** (*any*):
  the compression desired. If not given it will be autodetected from the
  filename.


### `chmp.ds.clear_tqdm`
`chmp.ds.clear_tqdm()`

Close any open TQDM instances to prevent display errors


### `chmp.ds.sha1`
`chmp.ds.sha1(obj)`

Create a hash for a json-encode-able object


### `chmp.ds.random`
`chmp.ds.random(obj)`

Return a random float in the range [0, 1)


### `chmp.ds.np_seed`
`chmp.ds.np_seed(obj)`

Return a seed usable by numpy.


### `chmp.ds.tf_seed`
`chmp.ds.tf_seed(obj)`

Return a seed usable by tensorflow.


### `chmp.ds.std_seed`
`chmp.ds.std_seed(obj)`

Return a seed usable by python random module.


### `chmp.ds.shuffle`
`chmp.ds.shuffle(obj, l)`

Shuffle `l` in place using Fisher–Yates algorithm.

See: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle


### `chmp.ds.timeshift_index`
`chmp.ds.timeshift_index(obj, dt)`

Return a shallow copy of `obj` with its datetime index shifted by `dt`.


### `chmp.ds.to_start_of_day`
`chmp.ds.to_start_of_day(s)`

Return the start of the day for the datetime given in `s`.


### `chmp.ds.to_time_in_day`
`chmp.ds.to_time_in_day(s, unit=None)`

Return the timediff relative to the start of the day of `s`.


### `chmp.ds.to_start_of_week`
`chmp.ds.to_start_of_week(s)`

Return the start of the week for the datetime given `s`.


### `chmp.ds.to_time_in_week`
`chmp.ds.to_time_in_week(s, unit=None)`

Return the timedelta relative to weekstart for the datetime given in `s`.


### `chmp.ds.to_start_of_year`
`chmp.ds.to_start_of_year(s)`

Return the start of the year for the datetime given in `s`.


### `chmp.ds.to_time_in_year`
`chmp.ds.to_time_in_year(s, unit=None)`

Return the timediff relative to the start of the year for `s`.

