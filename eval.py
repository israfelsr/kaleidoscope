# Run evaluations, experiments and plots here

import argparse
import pandas as pd
import os
from datasets import load_dataset, Dataset, load_from_disk
from huggingface_hub import login
from eval_utils import (
    EVALUATION_STYLES,
    perform_complete_evaluation,
    perform_metrics,
    perform_descriptive_statistics,
    perform_experiments,
    perform_plots,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=str,
        default="all",
        help="number of samples to evaluate",
    )
    parser.add_argument("--results", type=str, required=True, help="inference output")
    parser.add_argument(
        "--is_hf_dataset",
        type=str,
        required=True,
        help="boolean to determine if it is a name/path from a dataset in HF.",
    )
    parser.add_argument(
        "--hf_token", type=str, help="HF token to access private datasets (if needed)"
    )
    parser.add_argument(
        "--evaluation_style",
        type=str,
        required=True,
        help=f"type of evaluation to perform. Should be one of: {', '.join(EVALUATION_STYLES)}",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="where to save output results"
    )
    parser.add_argument(
        "--filter_data_by",
        type=str,
        default=None,
        help="how to filter the entry dataset",
    )
    args = parser.parse_args()
    return args


def run_evaluation(results, style, output_folder):

    if style not in EVALUATION_STYLES:
        raise NameError(
            f"{style} is not a supported evaluation style. Evaluation styles: {EVALUATION_STYLES}"
        )

    os.makedirs(output_folder, exist_ok=True)

    if style == "complete":
        perform_complete_evaluation(results, output_folder)
    if style == "metrics":
        perform_metrics(results, output_folder)
    if style == "statistics":
        perform_descriptive_statistics(results, output_folder)
    if style == "experiments":
        perform_experiments(results, output_folder)
    if style == "plotting":
        perform_plots(results, output_folder)


def load_dataset_from_entry(args):

    if args.hf_token:
        print("Logging in...")
        login(args.hf_token)

    if isinstance(args.results, str):
        if args.is_hf_dataset == "True":
            results = load_from_disk(args.results)["train"]
            results_df = results.to_pandas()
        else:
            results_df = pd.read_json(args.results)
    else:
        raise TypeError(f"Unexpected dataset format: {args.results}")

    if args.num_samples != "all":
        results_df = results_df.head(int(args.num_samples))

    return results_df


def main():
    args = parse_args()

    # Load dataset.
    results = load_dataset_from_entry(args)
    run_evaluation(results, args.evaluation_style, args.output_folder)


if __name__ == "__main__":
    main()
