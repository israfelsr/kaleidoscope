import argparse
import json
import re
from tqdm import tqdm
import ast


MAP_NON_LATIN = {
    "Б": "B",  # Ukrainian
    "Ц": "C",  # Ukrainian
    "Д": "D",  # Ukrainian
    "أ": "A",  # Arabic (Alif)
    "ب": "B",  # Arabic (Ba)
    "ج": "C",  # Arabic (Jeem)
    "د": "D",  # Arabic (Dal)
    "Г": "D",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="path to raw inference",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to formatted inference",
    )
    args = parser.parse_args()
    return args


def extract_choice_dict(s):
    # if the model outputs only the choice
    if len(s) == 1:
        return s
    match = re.search(r'\{\s*"choice":\s*.*?\s*\}', s)
    if match:
        try:
            return ast.literal_eval(match.group())  # Safely convert string to dict
        except Exception as e:
            print(f"Error parsing choice: {match}")
    return None


def map_to_choice(json_data):
    if isinstance(json_data, str) and len(json_data) == 1:
        choice = json_data
    else:
        choice = json_data.get("choice", "").strip().upper()
    if len(choice) > 1:
        individual_choices = [c.strip() for c in choice.split(".")]
        valid_choices = [c for c in individual_choices if c in {"A", "B", "C", "D"}]
        choice = valid_choices[0] if len(valid_choices) == 1 else ""
    choice = MAP_NON_LATIN.get(choice, choice)
    return choice if len(choice) == 1 and choice in {"A", "B", "C", "D"} else None


def extract_choice(output):
    json_str = output.strip("```json\n").strip("\n```")
    try:
        # Parse the JSON string
        json_data = json.loads(json_str)
        # Check if "choice" key exists and its value is a single character (A, B, C, D, etc.)
        if "choice" in json_data and isinstance(json_data["choice"], str):
            # Normalize the value (e.g., strip whitespace, convert to uppercase)
            choice = json_data["choice"].strip().upper()
            # Check if the choice is a single character or a clear value
            if len(choice) == 1 and choice in {"A", "B", "C", "D"}:
                return choice
    except json.JSONDecodeError:
        print(output)
    return None  # Invalid or unclear answer


def main():
    args = parse_args()
    with open(args.results_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"Number of inference output: {len(data)}")
    prediction_field = next(
        (key for key in data[0].keys() if key.startswith("prediction_by_")),
        "prediction",
    )

    def get_missing_answers(data):
        missing_answers = []
        for sample in data:
            if sample[prediction_field] not in [0, 1, 2, 3]:
                missing_answers.append(sample)
        return missing_answers

    # First pass
    for sample in data:
        answer = sample["reasoning"].split("assistant\n")[-1]
        dictionary = extract_choice_dict(answer)
        if dictionary:
            choice = map_to_choice(dictionary)
            if choice:
                sample["prediction"] = ord(choice) - ord("A")
            else:
                pass
                # print(choice)
        else:
            pass
            # print(sample["reasoning"])
    missing = get_missing_answers(data)
    print(f"Number of missing answers: {len(missing)}")

    import code

    code.interact(local=locals())
    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(data, f, indent=2)  #

        print(f"Formatted data saved to: {args.save_path}")


if __name__ == "__main__":
    main()
