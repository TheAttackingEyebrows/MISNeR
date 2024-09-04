import json
import os
import torch
import pdb


specifications_filename = "specs.json"


def load_experiment_specifications(experiment_directory):

    filename = experiment_directory

    if not os.path.isfile(filename):
        raise Exception(
            "The experiment directory ({}) does not include specifications file "
            + '"specs.json"'.format(experiment_directory)
        )

    return json.load(open(filename))

specs = load_experiment_specifications("/mnt/HDD1/Gary/meshSDF/splits/only_pelvis_train.json")

data_source = specs["DataSource"]
