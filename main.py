from os import getcwd

import dataReader.dataReader
from dataReader.dataReader import Reader
from Models import discourseIndicators, SVM_pipeline


# sorting out what method to take
def main() -> None:
    """
    Start of Implementation of methods
    """
    print("Please enter method type:\n> DI - Discourse Indicators\n> SVM\n> CM - Combined Method (uses all "
          "above)")
    # method_type = input().lower()
    method_type = "cm"

    # see if value appropriate
    allowed = {"di", "svm", "cm"}
    if method_type not in allowed:
        raise ValueError(f"{method_type} is not a correct value",
                         "Please restart the program and enter a correct value")

    # load in the data
    if method_type == "di":
        discourseIndicators.run()
    elif method_type == "svm":
        SVM_pipeline.run()
    elif method_type == "cm":
        print("Combined Method")
        SVM_pipeline.run_combined()
    else:
        print("Option not selcted please restart the program")


# Start script
if __name__ == '__main__':
    path = getcwd() + "/ArgumentAnnotatedEssays-2.0/"
    main()
