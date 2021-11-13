# name ethnicity classification

## :arrow_down: installation:

- repository installation:
    ```
    git clone https://github.com/hollowcodes/name-ethnicity-classification.git
    cd name-ethnicity-classifier/
    ```
- dependencies: Python>=3.7, PyTorch, NumPy, Pandas

---

## 👨‍💻 usage:

 - ## :grey_exclamation: read this first:

    Before you start classifying, check out the different model configurations inside the folder [model_configurations/](./model_configurations/) or in the table below.

    There you will find different models which each classify a unique set of nationalities.

    The README.md in each model folder will inform you about which ethnicities it can classify, its performance and more information you should know about it.

    When using this console interface, you can specify which model you want to use.

    ---

 - ## :round_pushpin: classifying names in a given .csv file :

    ### :heavy_dollar_sign: example command:
    ```
    python predict_ethnicity.py -i .\examples\names.csv -o .\examples\predicted_ethnicities.csv -m 21_nationalities_and_else -d gpu -b 64
    ```

    ### :black_flag: flags:
    | flag | description | option |
    | :------------- |:------------- | ----- |
    | ```-i, --input``` | path to .csv containing (first and last) names; must contain one column called "names" (file-name freely selectable) | optional, alternative: -n | 
    | ```-o, --output``` | path to .csv in which the names along with the predictions will be stored (file will be created if it doesn't exist; file-name freely selectable) | required after -i |
    | ```-m, --model``` | name of model configuration which can be chosen from "model_configurations/" or from the table below | optional, default: 21_nationalities_and_else |
    | ```-d, --device``` | device on which the model will run, must be either "gpu" or "cpu" | optional, default: gpu if CUDA detected |
    | ```-b, --batchsize``` | specifies how many names will be processed in parallel (if it crashes choose a batch-size smaller than the amount of names in your .csv file) | optional, default: amount of names in input-file |

    ### :page_facing_up: example files:
    "names.csv" has to have one column named "names" (upper-/ lower case doesn't matter):
    ```csv
    1 names,
    2 John Doe,
    3 Max Mustermann,
    ```

    After running the command, the "predictions.csv" will look like this:
    ```csv
    1 names,ethnicities
    2 John Doe,american
    3 Max Mustermann,german
    ```

    ---

 - ## :round_pushpin: predicting a single name:

    ### :heavy_dollar_sign: example command:
    ```
    python3 predict_ethnicitiy.py -n "Gonzalo Rodriguez"

    >> name: Gonzalo Rodriguez - predicted ethnicity: spanish
    ```

    ### :black_flag: flags:
    | flag | description | option |
    | :-------------: |:------------- | ----- |
    | ```-n, --name``` | first and last name (upper-/ lower case doesn't matter) | optional, alternative: -i | 
    | ```-m, --model``` | name of model configuration which can be chosen from "model_configurations/" or from the table below | optional, default: 21_nationalities_and_else |

---

## :earth_africa: models:

| name | nationalities/groups | accuracy |
| ------------- |:------------- | :-----:|
| ```28_nationalities_english_once``` | <details><summary>click to see nationalities</summary>british, norwegian, indian, hungarian, spanish, german, zimbabwean, portugese, polish, bulgarian, bangladeshi, turkish, belgian, pakistani, italian, romanian, lithuanian, french, chinese, swedish, nigerian, greek, south african, japanese, dutch, danish, russian, filipino</details> | 78.54% |
| ```21_nationalities_and_else``` |<details><summary>click to see nationalities</summary>british, else, indian, hungarian, spanish, german, zimbabwean, polish, bulgarian, turkish, pakistani, italian, romanian, french, chinese, swedish, nigerian, greek, japanese, dutch, ukrainian, danish, russian</details> | 81.08% |
| ```8_groups``` | <details><summary>click to see nationalities</summary>african, celtic, eastAsian, european, hispanic, muslim, nordic, southAsian</details> | 83.55% |
| ```chinese_and_else``` | <details><summary>click to see nationalities</summary>chinese, else</details> | 98.55% |
| ```20_most_occuring_nationalities``` | <details><summary>click to see nationalities</summary>british, norwegian, indian, irish, spanish, american, german, polish, bulgarian, turkish, pakistani, italian, romanian, french, australian, chinese, swedish, nigerian, dutch, filipin</details> | 75.36% |







