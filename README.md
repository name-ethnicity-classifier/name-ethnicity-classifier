# Name Ethnicity Classification

This repository is part of the [name-to-ethnicity](https://www.name-to-ethnicity.com) research project. If you use this classifier for your scientific work, please cite our [paper](https://link.springer.com/article/10.1007/s00146-022-01619-4#citeas).


## What is name-ethnicity classification?
Name-ethnicity classification is the process of using a person's name to predict their ethnicity. It involves analyzing the linguistic features of the name to determine the likely ethnicity. This can help researchers detect potential biases and discrimination in various contexts, such as education, employment, housing, and healthcare.


## :arrow_down: Installation:
### Get repository:
```
git clone https://github.com/name-ethnicity-classifier/name-ethnicity-classifier
cd name-ethnicity-classifier/
```
### Dependencies:
Install the followng packages via ``pip`` or ``conda``:
``Python>=3.7``, ``PyTorch``, ``NumPy``, ``Pandas``

---

## 👨‍💻 Usage:

### Read this first:

Before you start classifying, check out the different model configurations inside the folder [model_configurations/](./model_configurations/) or in the table below.

There you will find different models which each classify a unique set of nationalities.

The README.md in each model folder will inform you about which ethnicities it can classify, its performance and more information you should know about it.

When using this console interface, you can specify which model you want to use.

##### None of the models is suitable for your problem?
On our website, www.name-to-ethnicity.com, you can request custom models trained on selected ethnicities (for free!).

### Command line flags:
| flag | description | example |
| :------------- |:------------- | ----- |
| ```-i, --input``` | Sets the path to an input .csv file containing first and last names; must contain one column called "names". | ``-i "./examples/name.csv"`` (required unless ``-n`` is used) | 
| ```-o, --output``` | Path to an output .csv in which the names along with the predictions will be stored (file will be created if it doesn't exist). | ``-o "./examples/predictions.csv"`` (optional, default: ``{input file name}_output.csv``) |
| ```-m, --model``` | Name of model configuration which can be chosen from "model_configurations/" or from the table below. | ``-m indian_and_else`` (optional, default: ``21_nationalities_and_else``) |
| ```-d, --device``` | Device on which the model will run, must be either "gpu" or "cpu". | ``-m "gpu"`` (optional, default: ``gpu``) |
| ```-b, --batchsize``` | Specifies how many names will be processed in parallel (if it crashes choose a batch-size smaller than the amount of names in your .csv file). | ``-b 128`` (optional, default: amount of names in input-file) |
| ```--distribution``` | If set, the output with contain the entire output distribution, ie. providing the confidence for all possible ethnicities. | No parameter |
| ```-n, --name``` | Alternative to ``-i``, expects just a single name which is then predicted | ``-n "cixin liu"`` (required unless ``-i`` is used) | 

---

### Option 1: Classifying names in a given .csv file :
#### Example command:
```
python predict_ethnicity.py -i ./examples/names.csv -o ./examples/predicted_ethnicities.csv -m 21_nationalities_and_else -d gpu -b 64
```
#### Example files:
The input .csv file has to have one column named "names" (upper-/ lower case doesn't matter):
| names           |
|-----------------|
| John Doe        |
| Max Mustermann  |

After running the command, the output .csv will look like this:
| names           | predictions | confidences |
|-----------------|-------------|-------------|
| John Doe        | american    | 0.73        |
| Max Mustermann  | german      | 0.92        |

If the ``--distribution`` flag was set the output .csv will look like this:
| names           | predictions | american | german |
|-----------------|-------------|----------|--------|
| John Doe        | american    | 0.73     | 0.27   |
| Max Mustermann  | german      | 0.08     | 0.92   |

---

### Option 2: Predicting a single name:

#### Example command:
```
python3 predict_ethnicitiy.py -n "Gonzalo Rodriguez"

>> name: Gonzalo Rodriguez - predicted ethnicity: spanish
```

---

## :earth_africa: Models:

| name | nationalities/groups | accuracy |
| ------------- |:------------- | :-----:|
| ```28_nationalities_english_once``` | <details><summary>click to see nationalities</summary>``british`` ``norwegian`` ``indian`` ``hungarian`` ``spanish`` ``german`` ``zimbabwean`` ``portugese`` ``polish`` ``bulgarian`` ``bangladeshi`` ``turkish`` ``belgian`` ``pakistani`` ``italian`` ``romanian`` ``lithuanian`` ``french`` ``chinese`` ``swedish`` ``nigerian`` ``greek`` ``south african`` ``japanese`` ``dutch`` ``danish`` ``russian`` ``filipino``</details> | 78.54% |
| ```21_nationalities_and_else``` |<details><summary>click to see nationalities</summary>``british`` ``else`` ``indian`` ``hungarian`` ``spanish`` ``german`` ``zimbabwean`` ``polish`` ``bulgarian`` ``turkish`` ``pakistani`` ``italian`` ``romanian`` ``french`` ``chinese`` ``swedish`` ``nigerian`` ``greek`` ``japanese`` ``dutch`` ``ukrainian`` ``danish`` ``russian``</details> | 81.08% |
| ```8_groups``` | <details><summary>click to see nationalities</summary>``african`` ``celtic`` ``eastAsian`` ``european`` ``hispanic`` ``muslim`` ``nordic`` ``southAsian``</details> | 83.55% |
| ```chinese_and_else``` | <details><summary>click to see nationalities</summary>``chinese`` ``else``</details> | 98.55% |
| ```20_most_occuring_nationalities``` | <details><summary>click to see nationalities</summary>``british`` ``norwegian`` ``indian`` ``irish`` ``spanish`` ``american`` ``german`` ``polish`` ``bulgarian`` ``turkish`` ``pakistani`` ``italian`` ``romanian`` ``french`` ``australian`` ``chinese`` ``swedish`` ``nigerian`` ``dutch`` ``filipin``</details> | 75.36% |
| ```german_austrian_and_else``` | <details><summary>click to see nationalities</summary>``german/austrian combined`` ``else``</details> | 88.1% |
| ```indian_and_else``` | <details><summary>click to see nationalities</summary>``indian`` ``else``</details> | 94.63% |







