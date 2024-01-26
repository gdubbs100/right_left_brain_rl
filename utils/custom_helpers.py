import argparse
import json

def get_args_from_config(filepath):

    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
        
    with open(filepath + '/config.json', 'rt') as f:
            args.__dict__.update(json.load(f))

    return args