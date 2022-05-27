from os import getcwd


# sorting out what method to take
def main(path: str) -> None:
    """
    Start of Implementation of methods
    :param path_to_file:
    """
    print("Please enter method type:\n> TS - Topic Similarity\n> SVM\n> TBD\n> CM - Combined Method (uses all above)")
    method_type = input().lower()

    # see if value appropriate
    allowed = {"ts", "svm", "tbd", "cm"}
    if method_type not in allowed:
        raise ValueError(f"{method_type} is not a correct value",
                         "Please restart the program and enter a correct value")
    # load in the data

    if method_type == "ts":
        pass
    elif method_type == "svm":
        pass
    elif method_type == "tbd":
        pass
    elif method_type == "cm":
        pass


# Start script
if __name__ == '__main__':
    path = getcwd() + "/ArgumentAnnotatedEssays-2.0/"
    main(path)
