import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("device")
    parser.add_argument("type")
    return parser.parse_args()