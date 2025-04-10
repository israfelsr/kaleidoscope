import argparse
import numpy as np
from datasets import load_from_disk, Dataset
import os
import random
import json
from tqdm import tqdm
from collections import defaultdict

from model_utils import (
    initialize_model,
    query_model,
    generate_prompt,
    SUPPORTED_MODELS,
)

IMAGE_ROOT = "./"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="number of samples to test",
    )
    parser.add_argument(
        "--method",
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Pass the file of aswers from where to resume",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="Optional extra output name",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="",
        help="select a subset of the dataset [multimoda, text-only].",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    args = parser.parse_args()
    return args


def map_image_path(example):
    if example["image"] is not None:
        example["image"] = IMAGE_ROOT + example["image"]
    return example


def filter_ready(dataset, results):
    hf_questions = set(
        (
            example["question"],
            example["file_name"],
            example["image_png"],
            example["source"],
            example["original_question_num"],
        )
        for example in dataset
    )
    json_questions = set(
        (
            example["question"],
            example["file_name"],
            example["image_png"],
            example["source"],
            example["original_question_num"],
        )
        for example in results
    )
    # Find duplicates
    common_questions = hf_questions.intersection(json_questions)

    # Filter out duplicates from the HF dataset
    filtered_data = [
        example
        for example in dataset
        if (
            example["question"],
            example["file_name"],
            example["image_png"],
            example["source"],
            example["original_question_num"],
        )
        not in common_questions
    ]

    # Convert back to Hugging Face dataset
    return Dataset.from_list(filtered_data)


def load_and_filter_dataset(
    dataset_name: str,
    lang: str,
    num_samples: int,
    method: str,
    subset: str,
    results: list = [],
):
    """
    Load and filter the dataset based on language and number of samples.
    """
    # TODO: ADD OTHER FILTERS
    dataset = load_from_disk(dataset_name)
    dataset = dataset.map(map_image_path)
    few_shot_examples = defaultdict(list)
    if method == "few-shot":
        assert len(dataset) == 2
        for sample in dataset["train"]:
            lang = sample["language"]
            few_shot_examples[lang].append(sample)
        dataset = dataset["test"]
    if results:
        dataset = filter_ready(dataset, results)
    if num_samples is not None:
        dataset = dataset.select(range(num_samples))
    if subset == "multimodal":
        dataset = dataset.filter(lambda sample: sample["image"] is not None)
    return dataset, few_shot_examples


def evaluate_model(args):
    """
    Run the evaluation pipeline for the specified model.
    """
    # Set path
    if args.resume:
        output_path = args.resume
        with open(output_path, "r") as f:
            results = json.load(f)
        unique_results = set(
            (
                example["question"],
                example["file_name"],
                example["image_png"],
                example["source"],
                example["original_question_num"],
            )
            for example in results
        )
        results = [
            example
            for example in results
            if (
                example["question"],
                example["file_name"],
                example["image_png"],
                example["source"],
                example["original_question_num"],
            )
            in unique_results
        ]
    else:
        output_folder = f"outputs/{args.method}/model_{args.model}"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"results{args.output_name}.json")
        results = []

    # Initialize model
    model, processor = initialize_model(
        args.model, args.model_path, args.api_key, args.ngpu
    )

    print(f"Model loaded from {args.model}")

    # Load dataset
    dataset, few_shot_samples = load_and_filter_dataset(
        args.dataset,
        args.selected_langs,
        args.num_samples,
        args.method,
        args.subset,
        results,
    )
    print(dataset)

    # Evaluate each question
    for t, question in tqdm(enumerate(dataset), total=len(dataset)):
        lang = question["language"]
        # Generate prompt. Note that only local models will need image_paths separatedly.

        prompt, image_paths = generate_prompt(
            model_name=args.model,
            question=question,
            lang=lang,
            few_shot_samples=few_shot_samples,
            method=args.method,
        )

        # Query model
        reasoning, prediction = query_model(
            args.model,
            model,
            processor,
            prompt,
            image_paths,
        )
        question["prediction"] = prediction
        question["reasoning"] = reasoning
        question["prompt_used"] = prompt
        result_metadata = question.copy()
        results.append(result_metadata)

        if (t + 1) % 100 == 0 or (t + 1) == len(dataset):
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Ongoing {t} results saved to {output_path}")

    # Save results to file
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
