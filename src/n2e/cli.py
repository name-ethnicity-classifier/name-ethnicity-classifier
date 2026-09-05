
from n2e.predict import *

def get_flags() -> Union[list, bool, str, str, int, str]:
    """
    Handles console arguments

    :return list: list of names to predict ethnicities
    :return bool: wether the user wants the entire output distribution
    :return list: list of names to predict ethnicities
    :return str: path of csv-file in which to save ethnicities
    :return str: model configuration name
    :return int: batch-size for forward pass
    :return str: host device for the model
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=False, help="Path to .csv containing (first and last) names; must contain one column called 'names' (name freely selectable)")
    parser.add_argument("-o", "--output", required=False, help="Path to .csv in which the names along with the predictions will be stores (file will be created if it doesn't exist; name freely selectable)")
    parser.add_argument("-d", "--device", required=False, help="Must be either 'gpu' or 'cpu' (standard: 'gpu' if cuda support is detected, else 'cpu')")
    parser.add_argument("-b", "--batchsize", required=False, help="Specifies how many names will be processed in parallel (standard: process all names in parallel; if it crashes choose a batch-size smaller than the amount of names in your .csv file; the bigger the batchsize the faster it will classify the names)")
    parser.add_argument("-n", "--name", required=False, help="First and last name (upper-/ lower case doesn't matter)")
    parser.add_argument("-m", "--model", required=False, help="Folder name of model configuration which can be chosen from 'model_configurations/' (standard: '21_nationalities_and_else')")
    parser.add_argument("--distribution", required=False, action="store_true", help="If set, the entire output distribution is returned")

    args = vars(parser.parse_args())

    # check if -/--name is used and -i/--input not
    if args["name"] != None and args["input"] == None:
        names = [args["name"]]
        csv_out_path = None
        get_distribution = False
    
    # check if -/--name is not used but -i/--input is
    elif args["name"] == None and args["input"] != None:
        csv_in_path = args["input"]
        csv_out_path = args["output"]
        names = pd.read_csv(csv_in_path)["names"].tolist()

    # check if -/--name and -c/--csv are both not used (raise error)
    elif args["name"] == None and args["input"] == None:
        raise ValueError("Either -n/--name or -i/--input must be set!")

    # check if -/--name and -c/--csv are both used (raise error)
    elif args["name"] != None and args["input"] != None:
        raise ValueError("-n/--name and -i/--input can't both be set!")

    # create an output file name if none was specified
    if args["input"] != None and args["output"] == None:
        csv_out_path = f"{args['input'].removesuffix('.csv')}_output.csv"

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

    # check if the -i/--input and -o/--output flag was set, by checking if there is a csv-save-file, if so: save names with their ethnicities
    if csv_out_path != None:
        result_df.to_csv(csv_out_path, index=False)

        print("\nClassified all names and saved to {}.\n".format(csv_out_path))

    # if a single name was parsed using -n/--name, print the predicition
    else:
        print("\nname: {} - predicted ethnicity: {}".format(result_df["names"][0], result_df["predictions"][0]))

if __name__ == "__main__":
    main()