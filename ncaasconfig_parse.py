import os
import yaml
import argparse
import sys

MANDATORY_FILES = ["data", "compute", "model", "training", "params"]

FILE_TYPES = {
    "data": "hdf5",
    "architecture": "json",
    "compute": "json",
    "model": "json",
    "training": "json",
    "params": "json",
    "sessions": "csv"
}

COMMAND_FLAGS = {
    "compute": "--compute_config",
    "model": "--model_config",
    "training": "--training_config",
    "params": "--data_config"
}


def yaml_to_dict(filename):
    try:
        with open(filename) as f:
            dictionary = yaml.full_load(f)
        return dictionary
    except:
        sys.exit("failed to load {}".format(filename))


def formatted_print(metadict):
    formatted = []
    for filetype, name in metadict.items():
        if name and not name.isspace() and name != "":
            formatted.append("{}:{}".format(filetype, name))

    print(formatted)


def check_data(metadata):
    for mandatory in MANDATORY_FILES:
        if mandatory not in metadata or not metadata[mandatory]:
            sys.exit("missing mandatory file {}".format(mandatory))

    for filetype, filename in metadata.items():
        if filename:
            ext = os.path.splitext(filename)[1]
            try:
                if ((type(FILE_TYPES[filetype]) == str and ext != FILE_TYPES[filetype]) or
                        (type(FILE_TYPES[filetype]) == list and ext not in FILE_TYPES[filetype])):
                    if filetype not in MANDATORY_FILES and filename == "":
                        continue
                    else:
                        sys.exit("{} is of an invalid file type for {} - should be {}".format(
                            filename, filetype, FILE_TYPES[filetype]
                        ))
            except KeyError:
                sys.exit("{} isn\'t supposed to be in here!".format(filetype))

    formatted_print(metadata)


def main(args):
    metafile = args.meta[0]
    metadata = yaml_to_dict(metafile)
    check_data(metadata)
    formatted_print(metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "meta", help="meta.json with path", nargs=1)
    args = parser.parse_args()

    main(args)
