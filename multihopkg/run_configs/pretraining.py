import argparse

def get_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser()

    # Data Processing
    ap.add_argument("--pretrained_sun_model_loc", type=str, default="models/sun")
    ap.add_argument("--data_dir", type=str, default="./data/FB15k")

    # Hardware
    ap.add_argument("--device", type=str, default="cuda:0")

    # TODO: We may later add something like ../../configs/my_config.yaml but for pretraining here. Or just simplify this bunch

    return ap.parse_args()
