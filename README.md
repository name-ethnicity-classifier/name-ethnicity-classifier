# Name Ethnicity Classification

This repository is part of the [name-to-ethnicity](https://www.name-to-ethnicity.com) research project. If you use this classifier for your scientific work, please cite our [paper](https://link.springer.com/article/10.1007/s00146-022-01619-4#citeas).


## What is name-ethnicity classification?
Name-ethnicity classification is the process of using a person's name to predict their ethnicity. It involves analyzing the linguistic features of the name to determine the likely ethnicity. This can help researchers detect potential biases and discrimination in various contexts, such as education, employment and healthcare.

This project contains a Python package and CLI tool called `n2e`.

## :arrow_down: Installation:

This project uses the [uv](https://docs.astral.sh/uv/) Python package manager.

### Install the package: 

For installing the `n2e` CLI interface, run: 
```bash
uv tool install n2e
```

For the `n2e` Python package, run:
```bash
uv add n2e
```

If you don't use `uv`, run:
```bash
pip install n2e
```

### For local development:
If you want to set up this repository for local development, clone the repository instead:
```bash
git clone https://github.com/name-ethnicity-classifier/name-ethnicity-classifier
cd name-ethnicity-classifier/
```

### Dependencies:
The project was tested with **Python ≥ 3.10**. The following packages are installed and used as dependencies: ``NumPy``, ``Pandas`` and ``PyTorch``.


## 👨‍💻 Usage:

### Read this first:

Before you start classifying, check out the different model configurations in the "Models" table below. Models are hosted as GitHub release assets and downloaded automatically the first time you use one, then cached in `~/.cache/n2e/`.

There you will find different models which each classify a unique set of ethnicities and their average accuracy.

When using the CLI or the Python package, you can specify which model you want to use.

##### None of the models is suitable for your problem?
On the website [www.name-to-ethnicity.com](https://www.name-to-ethnicity.com) you can request custom models trained on selected ethnicities (for free!). Alternatively, you can use the [model-training](https://github.com/name-ethnicity-classifier/model-training) repository to train models locally.

### n2e CLI:

The following flags are available:
| flag | description | example |
| :------------- |:------------- | ----- |
| ```-i, --input``` | Sets the path to an input .csv file containing first and last names; must contain one column called "names". | ``-i "./examples/names.csv"`` (required unless ``-n`` is used) | 
| ```-o, --output``` | Path to an output .csv in which the names along with the predictions will be stored (file will be created if it doesn't exist). | ``-o "./examples/predictions.csv"`` (optional, default: ``{input file name}_output.csv``) |
| ```-m, --model``` | Name of model configuration which can be chosen from the table below. | ``-m indian_and_else`` (optional, default: ``21_nationalities_and_else``) |
| ```-d, --device``` | Device on which the model will run, must be either "gpu" or "cpu". | ``-d "gpu"`` (optional, default: ``gpu``) |
| ```-b, --batchsize``` | Specifies how many names will be processed in parallel (if it crashes choose a batch-size smaller than the amount of names in your .csv file). | ``-b 128`` (optional, default: amount of names in input-file) |
| ```--distribution``` | If set, the output will contain the entire output distribution, ie. providing the confidence for all possible ethnicities. | No parameter |
| ```-n, --name``` | Alternative to ``-i``, expects just a single name which is then predicted | ``-n "cixin liu"`` (required unless ``-i`` is used) | 

#### Option 1 - Bulk classification:

To classify a list of names in a given `.csv` file, see the following example command:
```bash
n2e -i ./examples/names.csv -o ./examples/predicted_ethnicities.csv -m 21_nationalities_and_else -d gpu -b 64
```

The input .csv file has to have one column named "names" (upper-/ lower case doesn't matter):
| names                |
|----------------------|
| Giorgos Papadopoulos |
| Max Mustermann       |

After running the command, the output `.csv` will look like this:
| names                | predictions | confidences |
|----------------------|-------------|-------------|
| Giorgos Papadopoulos | greek       | 0.73        |
| Max Mustermann       | german      | 0.92        |

If the ``--distribution`` flag was set the output `.csv` will look like this:
| names                | predictions | greek    | german |
|----------------------|-------------|----------|--------|
| Giorgos Papadopoulos | greek       | 0.73     | 0.27   |
| Max Mustermann       | german      | 0.08     | 0.92   |

---

#### Option 2 - Classifying a single name:

To quickly classify just a single name, run:
```bash
n2e -n "Max Mustermann"

>> name: Max Mustermann - predicted ethnicity: german
```
---

### Python API:

Import `predict_ethnicities` to use `n2e` directly as a Python module:

```python
from n2e import predict_ethnicities

predict_ethnicities(
    names,                              # list[str]
    batch_size=128,                     # names processed in parallel
    model="21_nationalities_and_else",  # any model from the table below
    get_distribution=False,             # return confidences for every ethnicity
)
```

Returns a list of `(ethnicity, confidence)` tuples, one per name:

```python
>>> predict_ethnicities(["Giorgos Papadopoulos"])
[('greek', 99.045)]

>>> predict_ethnicities(["Giorgos Papadopoulos", "Max Mustermann"])
[('greek', 99.045), ('german', 60.34)]
```

With `get_distribution=True` you get a full distribution per name:

```python
>>> predict_ethnicities(["Giorgos Papadopoulos"], get_distribution=True)
[{'british': 0.073, 'else': 0.046, 'indian': 0.008, ...}]
```

## :earth_africa: Models:

| name | classes | accuracy |
| ------------- |:------------- | :-----:|
| ```28_nationalities_english_once``` | <details><summary>click to see classes</summary>``british`` ``norwegian`` ``indian`` ``hungarian`` ``spanish`` ``german`` ``zimbabwean`` ``portugese`` ``polish`` ``bulgarian`` ``bangladeshi`` ``turkish`` ``belgian`` ``pakistani`` ``italian`` ``romanian`` ``lithuanian`` ``french`` ``chinese`` ``swedish`` ``nigerian`` ``greek`` ``south african`` ``japanese`` ``dutch`` ``danish`` ``russian`` ``filipino``</details> | 78.54% |
| ```21_nationalities_and_else``` |<details><summary>click to see classes</summary>``british`` ``else`` ``indian`` ``hungarian`` ``spanish`` ``german`` ``zimbabwean`` ``polish`` ``bulgarian`` ``turkish`` ``pakistani`` ``italian`` ``romanian`` ``french`` ``chinese`` ``swedish`` ``nigerian`` ``greek`` ``japanese`` ``dutch`` ``ukrainian`` ``danish`` ``russian``</details> | 81.08% |
| ```8_groups``` | <details><summary>click to see classes</summary>``african`` ``celtic`` ``eastAsian`` ``european`` ``hispanic`` ``muslim`` ``nordic`` ``southAsian``</details> | 83.55% |
| ```chinese_and_else``` | <details><summary>click to see classes</summary>``chinese`` ``else``</details> | 98.55% |
| ```20_most_occuring_nationalities``` | <details><summary>click to see classes</summary>``british`` ``norwegian`` ``indian`` ``irish`` ``spanish`` ``american`` ``german`` ``polish`` ``bulgarian`` ``turkish`` ``pakistani`` ``italian`` ``romanian`` ``french`` ``australian`` ``chinese`` ``swedish`` ``nigerian`` ``dutch`` ``filipino``</details> | 75.36% |
| ```german_austrian_and_else``` | <details><summary>click to see classes</summary>``german/austrian combined`` ``else``</details> | 88.1% |
| ```indian_and_else``` | <details><summary>click to see classes</summary>``else`` ``indian``</details> | 94.63% |
| ```japanese_and_else``` | <details><summary>click to see classes</summary>``else`` ``japanese``</details> | 99.33% |
| ```newzealand_and_else``` | <details><summary>click to see classes</summary>``else`` ``new zealander``</details> | 66.71% |







