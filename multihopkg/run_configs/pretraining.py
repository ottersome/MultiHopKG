import argparse

def get_args() -> argparse.Namespace:

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_sun_model_loc", type=str, default="models/sun")

    return ap.parse_args()
