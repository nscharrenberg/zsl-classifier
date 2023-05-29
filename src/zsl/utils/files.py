import os
import pandas as pd
from zsl.utils import log


def load_csv(path: str, class_header: str, verbose: bool = False):
    log(f"Reading \"Seen\" dataset...", verbose=verbose)

    if not os.path.isfile(path):
        log(f"The path \"{path}\" could NOT be found!", 50)
        return

    seen_data = pd.read_csv(path)

    X = seen_data.drop(class_header, axis=1)
    y = seen_data[class_header]

    return X, y
