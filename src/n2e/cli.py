
from n2e.predict import *

def get_flags() -> tuple:
    """
    Handles console arguments

    :return list: list of names to predict ethnicities
    :return bool: wether the user wants the entire output distribution
    :return str: path of csv-file in which to save ethnicities
    :return str: model configuration name
    :return int: batch-size for forward pass
    :return str: host device for the model
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=True, help="Path to .csv containing names; must contain one column called 'names'")
    parser.add_argument("-o", "--output", required=False, help="Path to .csv in which the names along with the predictions will be stored (file will be created if it doesn't exist)")
    parser.add_argument("-d", "--device", required=False, help="Must be either 'gpu' or 'cpu' (standard: 'gpu' if cuda support is detected, else 'cpu')")
    parser.add_argument("-b", "--batchsize", required=False, help="Specifies how many names will be processed in parallel (standard: process all names in parallel; if it crashes choose a batch-size smaller than the amount of names in your .csv file; the bigger the batchsize the faster it will classify the names)")
    parser.add_argument("-m", "--model", required=False, help="Name of the model configuration which can be chosen from the table in the README (standard: '21_nationalities_and_else')")
    parser.add_argument("--distribution", required=False, action="store_true", help="If set, the entire output distribution is returned")

    args = vars(parser.parse_args())

    input_df = pd.read_csv(args["input"])
    if "names" not in input_df.columns:
        raise ValueError("The input .csv must contain a column called 'names'.")

    names = input_df["names"].tolist()

    # create an output file name if none was specified
    if args["output"] == None:
        csv_out_path = f"{args['input'].removesuffix('.csv')}_output.csv"
    else:
        csv_out_path = args["output"]

    # check wether the user wants the entire output distribution
    get_distribution = args["distribution"]

    # get model
    if args["model"] == None:
        model = "21_nationalities_and_else"
    else:
        model = args["model"]

    # get batch-size
    if args["batchsize"] == None or int(args["batchsize"]) > len(names):
        batch_size = len(names)
    else:
        batch_size = int(args["batchsize"])

    # get device
    if args["device"] == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args["device"].lower() == "cpu":
        device = torch.device("cpu")
    elif args["device"].lower() == "gpu":
        if not torch.cuda.is_available():
            print("Couldn't find cuda on your system! Please use 'CPU' or install cuda when possible! Proceeding with CPU...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise NameError("Please use either 'GPU' or 'CPU' as device type!")

    return names, get_distribution, csv_out_path, model, batch_size, device


def main():
    # get names from console arguments
    names, get_distribution, csv_out_path, model, batch_size, device = get_flags()

    predictions = predict_ethnicities(names, batch_size, model, get_distribution)

    # stores either the entire output distribution in a dataframe or just the most likely ethnicity
    if get_distribution:
        result_df = pd.DataFrame(predictions)
        highest_confidence_ethnicities = result_df.idxmax(axis=1)
        result_df.insert(loc=0, column="names", value=names)
        result_df.insert(loc=1, column="predictions", value=highest_confidence_ethnicities)
    else:
        ethnicities, confidence = zip(*predictions)
        result_df = pd.DataFrame(list(zip(names, ethnicities, confidence)), columns=["names", "predictions", "confidences"])

    result_df.to_csv(csv_out_path, index=False)

    print("\nClassified all names and saved to {}.\n".format(csv_out_path))

if __name__ == "__main__":
    main()
