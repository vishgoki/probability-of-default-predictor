import argparse
from prediction import prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict financial defaults.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output CSV file")
    args = parser.parse_args()

    model_path = 'gam.pkl'

    prediction(args.input_csv, model_path, args.output_csv)

