from datasets import load_from_disk, Dataset, DatasetDict
import random
from collections import defaultdict


def main():
    dataset = load_from_disk("./dataset/stratified_dataset.hf")
    held_out_data = []
    remaining_data = []

    # Group dataset by (language, category)
    grouped_data = defaultdict(list)
    for example in dataset:
        # using text-only questions as few-shot for efficiency
        if example["image"] is None:
            key = example["language"]
            grouped_data[key].append(example)

    for key, questions in grouped_data.items():
        held_out = random.sample(questions, 3)  # Pick 3 random questions
        remaining = [q for q in questions if q not in held_out]

        held_out_data.extend(held_out)
        remaining_data.extend(remaining)

    # Convert back to Hugging Face datasets
    held_out_dataset = Dataset.from_list(held_out_data)
    remaining_dataset = Dataset.from_list(remaining_data)

    final_dataset = DatasetDict({"train": held_out_dataset, "test": remaining_dataset})
    final_dataset.save_to_disk("./dataset/dataset-fewshot.hf")
    print(f"Held out {len(held_out_data)} questions as few-shot (train).")
    print(f"Remaining dataset has {len(remaining_data)} questions (test).")


if __name__ == "__main__":
    main()
