## `chmp.csc`

Execution of scripts section by section.

Sometimes it may be helpful to run individual parts of a script inside an
interactive environment, for example Jupyter Notebooks. `CellScript` is
designed to support this use case. The basis are Pythn scripts with special cell
annotations. For example consider a script to define and train a model:

```python
#%% Setup
...

#%% Train
...

#%% Save
...
```

Where each of the `...` stands for arbitrary user defined code. Using
`CellScript` this script can be step by step as:

```python
script = CellScript("external_script.py")

script.run("Setup")
script.run("Train")
script.run("Save")
```

To list all available cells use `script.list()`.

The variables defined inside the script can be accessed and modified using the
`ns` attribute of the script. One example would be to define a parameter cell
with default parameters and the overwrite the values before executing the
remaining cells. Assume the script defines a parameter cell as follows:

```python
#%% Parameters
hidden_units = 128
activation = 'relu'
```

Then the parameters can be modified as in:

```python
script.run("Parameters")
script.ns.hidden_units = 64
script.ns.activation = 'sigmoid'
```

Beyond direct modification of the script namespace `CellScript` offers
different ways to interact with the script namespace. With
`script.assign` variables inside the script can be overwritten. For example,
the assignment from before could also have been written as:

```python
script.assign(hidden_units=64, activation='sigmoid')
```

Script objects can also export variables into the `__main__` module, which is
the namespace for example for Jupyter Notebooks. Exports can be declared with:

```python
script.export("loss_history", "model")
```

After exports are declared, the variables are copied from the script namespace
into the export namespace after each call to `run`.


### `chmp.csc.CellScript`
`chmp.csc.CellScript(path, *, cell_marker='%%', verbose=True, ns=None, export_ns=None)`

Allow to execute a python script step by step

`CellScript` is designed to be used inside Jupyter notebooks and allows to
execute an external script with special cell annotations cell by cell. The
script is reloaded before execution, but the namespace is persisted on this
`CellScript` instance.

The namespace of the script is available via the `ns` attribute:

```python
train_script("Setup")
print("parameters:", sorted(train_script.ns.model.state_dict()))

train_script("Train")
train_script.ns.model
```

#### Parameters

* **path** (*any*):
  The path of the script, can be a string or a [pathlib.Path](#pathlibpath).
* **cell_marker** (*any*):
  The cell marker used. Cells are defined as `# {CELL_MARKER} {NAME}`,
  with an arbitrary number of spaces allowed.
* **verbose** (*any*):
  If True, print a summary of the code executed for each cell.
* **ns** (*any*):
  The namespace to use for the execution. Per default a new module will
  be constructed. To share the same namespace with the currently running
  notebook it can be set to the `__main__` module.
* **export_ns** (*any*):
  The namespace to use for variable exports, see also the `export`
  method. Per default the `__main__` module will be used.


#### `chmp.csc.CellScript.run`
`chmp.csc.CellScript.run(*cells)`

Execute cells inside the script

#### Parameters

* **cells** (*any*):
  The cells to execute. They can be passed as the index of the cell
  or its name. Cell names only have to match the beginning of the
  name, as long as the prefix uniquely defines a cell. For example,
  instead of `"Setup training"` also `"Setup"` can be used.


#### `chmp.csc.CellScript.list`
`chmp.csc.CellScript.list()`

List the names for all cells inside the script.

If a cell is unnamed, `None` will be returned.


#### `chmp.csc.CellScript.get`
`chmp.csc.CellScript.get(cell)`

Get the source code of a cell

See the `run` method for details of what values can be used for the
cell parameter.


#### `chmp.csc.CellScript.assign`
`chmp.csc.CellScript.assign(**assignments)`

Assign variables in the script namespace.

#### Parameters

* **assignments** (*any*):
  values given as `variable=value` pairs.


#### `chmp.csc.CellScript.export`
`chmp.csc.CellScript.export(*names, **renames)`

Declare exports

#### Parameters

* **names** (*any*):
  variables which will be exported as-is into the export scope.
* **renames** (*any*):
  variables which will be renamed, when exported. For example, the
  declaration `script.export(script_model="model")` would export
  the `model`  variable in the script namespace as `script_model`
  into the export namespace.


#### `chmp.csc.CellScript.eval`
`chmp.csc.CellScript.eval(expr)`

Execute an expression inside the script namespace.

The expression can also be passed as a multiline string:

```python
result = script.eval('''
    a + b
''')
```


#### `chmp.csc.CellScript.exec`
`chmp.csc.CellScript.exec(source)`

Execute a Python block inside the script namespace.

The source is dedented to  allow using  `.eval` inside nested
blocks:

```python
if cond:
    script.exec('''
        hello = 'world'
    ''')
```


#### `chmp.csc.CellScript.load`
`chmp.csc.CellScript.load(cell)`

Load a cell into the notebook.

This function will replace the current notebook cell with the content
of the given script cell. The idea is to allow quick modification of
script code inside the notebook.

