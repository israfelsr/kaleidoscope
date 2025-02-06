import argparse
import numpy as np
from datasets import load_from_disk
import os
import random
import json
from tqdm import tqdm

from model_utils import (
    initialize_model,
    query_model,
    generate_prompt,
    fetch_system_message,
    SUPPORTED_MODELS,
    SYSTEM_MESSAGES,
    TEMPERATURE,
    MAX_TOKENS
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="number of samples to test",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="zero-shot",
        help="[few-shot, zero-shot]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--selected_langs",
        nargs="+",
        default=["all"],
        help="list of strings of language codes",
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="explicitly give an api key"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name or path"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=f"specify the model to use (must be one from the following: {', '.join(SUPPORTED_MODELS)})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model checkpoint or name",
    )
    args = parser.parse_args()
    return args


def load_and_filter_dataset(dataset_name: str, lang: str, num_samples: int):
    """
    Load and filter the dataset based on language and number of samples.
    """
    # TODO: ADD OTHER FILTERS
    dataset = load_from_disk(dataset_name)
    # Language
    # if lang != "all":
    #     dataset = dataset.filter(lambda sample: sample["language"] == lang)
    # else:
    #     print("evaluating all languages")
    # Level
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    return dataset


def evaluate_model(args):
    """
    Run the evaluation pipeline for the specified model.
    """
    # Initialize model
    model, processor = initialize_model(args.model, args.model_path, args.api_key)
    
    temperature = TEMPERATURE
    max_tokens = MAX_TOKENS
    system_message = fetch_system_message(SYSTEM_MESSAGES, lang)

    # Load dataset
    dataset = load_and_filter_dataset(
        args.dataset, args.selected_langs, args.num_samples
    )
    print(dataset)

    # Evaluate each question
    results = []
    for question in tqdm(dataset):
        lang = question["language"]
        # Generate prompt. Note that only local models will need image_paths separatedly.

        prompt, image_paths = generate_prompt(
            args.model,
            question,
            lang,
            system_message,
            args.setting
        )
        # Query model
        prediction = query_model(args.model,
                                 model, 
                                 processor, 
                                 prompt, 
                                 image_paths, 
                                 temperature=temperature,
                                 max_tokens=max_tokens
        )

        question["prediction_by_" + args.model] = prediction
        # question_json['prompt_used'] = prompt
        result_metadata = question.copy()
        results.append(result_metadata)

    # Save results to file
    output_folder = f"outputs/{args.setting}/mode_{args.model}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation completed. Results saved to {output_path}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    evaluate_model(args)


if __name__ == "__main__":
    main()
