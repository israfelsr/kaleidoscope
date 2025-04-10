import argparse
import json
from cohere import ClientV2
from tqdm import tqdm
from pathlib import Path


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
    parser.add_argument(
        "--pangea",
        type=bool,
        default=False,
        help="is pangea inference",
    )
    parser.add_argument(
        "--split",
        type=int,
        nargs=2,  # Expects exactly 2 values: start and end
        default=None,
        help="process a subset of data between start and end indices (0-based)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="output_name file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.results_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if args.split is not None:
        data = data[args.split[0] : args.split[1]]
    print(f"Number of inference output: {len(data)}")
    prediction_field = next(
        (key for key in data[0].keys() if key.startswith("prediction_by_")),
        "prediction",
    )
    with open("./results/pangea/pangea-LLM-formatted.json", "r") as f:
        ready = json.load(f)
    ready = set([int(list(sample.keys())[0]) for sample in ready])

    def get_missing_answers(data):
        missing_answers = []
        for t, sample in enumerate(data):
            if sample[prediction_field] not in [0, 1, 2, 3]:

                if t not in ready:
                    missing_answers.append((t, sample))
        return missing_answers

    missing = get_missing_answers(data)
    print(f"Number of missing answers: {len(missing)}")

    client = ClientV2(api_key="")

    llm_added = []

    for t, (i, sample) in tqdm(enumerate(missing), total=len(missing)):
        answer = sample.get("reasoning", "")

        if args.pangea:
            answer = answer.split("Answer: assistant ")[-1]

        response = client.chat(
            model="command-a-03-2025",
            messages=[
                {
                    "role": "system",
                    "content": 'You are an expert at extracting multiple-choice answers from unstructured text. Your task is to analyze the given text and identify the correct choice (A, B, C, D or None) based on the content. Follow these steps:\n\n1. Carefully read the text and look for any mention of a choice (A, B, C, or D).\n2. If the text explicitly states a choice (e.g., "The answer is B"), extract that choice.\n3. If the text uses non-Latin script (e.g., Arabic, Hindi, Cyrillic), map it to the corresponding Latin choice (A, B, C, D).\n4. If no explicit choice is found,output None.\n5. Always respond with in format {"choice": "<selection>"} with selection the exact choice (A, B, C, D or None). Do not include any additional text or explanation.\n',
                },
                {"role": "user", "content": [{"type": "text", "text": answer}]},
            ],
            temperature=0.7,
        )
        choice = response.message.content[0].text
        llm_added.append({i: choice})
        if (t + 1) % 100 == 0 or (t + 1) == len(missing):
            output_path = (
                Path(args.results_path).parent
                / f"{Path(args.results_path).parent.name}-LLM-formatted{args.output_name}.json"
            )
            with open(output_path, "w") as f:
                json.dump(llm_added, f, indent=2)

        #     if len(choice) == 1:
        #         sample[prediction_field] = ord(choice) - ord("A")
        #     else:
        #         sample[prediction_field] = choice
        # formatted_data.append(sample)
    # assert len(formatted_data) == len(data)
    # missing = get_missing_answers(formatted_data)
    print(f"Number of extracted: {len(llm_added)} out of {len(missing)}")
    import code

    code.interact(local=locals())
    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(llm_added, f, indent=2)
        print(f"Formatted data saved to: {args.save_path}")
    else:
        output_path = (
            Path(args.results_path).parent
            / f"{Path(args.results_path).parent.name}-LLM-formatted{args.output_name}.json"
        )
        with open(output_path, "w") as f:
            json.dump(llm_added, f, indent=2)
        print(f"Formatted data saved to: {output_path}")


if __name__ == "__main__":
    main()
