import argparse
import json
import re

NON_LATIN_TO_LATIN = {
    # Hindi (hi) - Devanagari script
    "hi": {
        "अ": "A",  # A
        "बी": "B",  # Ba
        "स": "C",  # Sa
        "ड": "D",  # Da
    },
    # Ukrainian (uk) - Cyrillic script
    "uk": {
        "А": "A",  # A
        "Б": "B",  # Be
        "В": "C",  # Ve (sometimes used as C)
        "Г": "D",  # Ge (sometimes used as D)
    },
    # Bengali (bn) - Bengali script
    "bn": {
        "অ": "A",  # A
        "ব": "B",  # Ba
        "স": "C",  # Sa
        "ড": "D",  # Da
    },
    # Telugu (te) - Telugu script
    "te": {
        "అ": "A",  # A
        "బ": "B",  # Ba
        "స": "C",  # Sa
        "డ": "D",  # Da
    },
    # Nepali (ne) - Devanagari script
    "ne": {
        "अ": "A",  # A
        "ब": "B",  # Ba
        "स": "C",  # Sa
        "ड": "D",  # Da
    },
    # Serbian (sr) - Cyrillic script
    "sr": {
        "А": "A",  # A
        "Б": "B",  # Be
        "В": "C",  # Ve (sometimes used as C)
        "Г": "D",  # Ge (sometimes used as D)
    },
    # Arabic (ar) - Arabic script
    "ar": {
        "ا": "A",  # Alif
        "ب": "B",  # Ba
        "ج": "C",  # Jeem
        "د": "D",  # Dal
    },
    # Russian (ru) - Cyrillic script
    "ru": {
        "А": "A",  # A
        "Б": "B",  # Be
        "В": "C",  # Ve (sometimes used as C)
        "Г": "D",  # Ge (sometimes used as D)
    },
    # Persian (fa) - Arabic script (Persian variant)
    "fa": {
        "ا": "A",  # Alif
        "ب": "B",  # Be
        "ج": "C",  # Jeem
        "د": "D",  # Dal
    },
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


def recategorize(dataset, category_map, category):
    with open(category_map, "r") as file:
        mapping = json.load(file)

    level_mapping = mapping.get(category, {})
    reverse_mapping = {}
    for new_category, old_categories in level_mapping.items():
        for old_category in old_categories:
            reverse_mapping[old_category] = new_category

    recategorized_dataset = []
    for item in dataset:
        original_category = item.get(category, "").strip()
        new_category = reverse_mapping.get(original_category, original_category)
        recategorized_item = item.copy()
        recategorized_item[category] = new_category
        recategorized_dataset.append(recategorized_item)

    return recategorized_dataset


def extract_answer(response):
    patterns = [
        r"\s*([abcd])\s*\)*",  # Latin script
        r"\s*([AБВГ])\s*\)*",
        r"\s*([अबीसड]).\s*\)*",  # Hindi (Devanagari)
        r"\s*([অবসড])\s*\)*",  # Bengali
        r"\s*([అబసడ])\s*\)*",  # Telugu
        r"\s*([ا ب ج د])\s*\)*",  # Arabic
    ]


def extract_choice(response):
    patterns = [
        r"<ANSWER>\s*([ABCD])\s*</ANSWER>",  # Matches <ANSWER>A</ANSWER>
        r"assistant\s+([ABCD])",  # Matches "assistant A"
        r"correct answer is\s*([ABCD])",  # Matches "correct answer is A"
        r"assistant\s*<\s*([ABCD])\s*",  # Matches "assistant <A"
        r"\n\n\s*([ABCD])",  # Matches "\n\n A"
        r"\n\n<ANSWER>\s*([ABCD])",  # Matches "\n\n<ANSWER>A"
        r"is option\s*([ABCD])",  # Matches "is option A"
        r"answer is:\s*([ABCD])",  # Matches "answer is: A"
        r"[Oo]ption\s*\(\s*([ABCD])\s*",  # Matches "Option (A)"
        r"([ABCD])\.",  # Matches "A." or "A "
        r"\*\*([ABCD])\*\*",
        r"\\boxed\{([ABCD])\}",  # Matches \boxed{B}"
        r"\n\n\*\*उत्तर:\s*([ABCD])\*\*",
        r":\s*([ABCD])",
        r"\s*\(([ABCD])\)",
        r"\n\n\*\*Option ([ABCD]):\*\*",
        r"es la opción ([ABCD])",
        r"\n\nइस प्रकार, सही उत्तर विकल्प ([ABCD]) है।",
    ]

    # Add patterns for non-Latin scripts
    for lang, mappings in NON_LATIN_TO_LATIN.items():
        for non_latin, latin in mappings.items():
            # Escape non-Latin characters for regex
            escaped_char = re.escape(non_latin)
            # Add patterns for non-Latin scripts
            patterns.extend(
                [
                    rf"<ANSWER>\s*({escaped_char})\s*</ANSWER>",  # Matches <ANSWER>अ</ANSWER>
                    rf"assistant\s+({escaped_char})",  # Matches "assistant अ"
                    rf"\*\*\s*({escaped_char})\s*[\.\)]",  # Matches "**अ) or **अ."
                    rf"assistant\s*<\s*({escaped_char})\s*",  # Matches "assistant <अ"
                    rf"\n\n\s*({escaped_char})",  # Matches "\n\n अ"
                    rf"\n\n<ANSWER>\s*({escaped_char})",  # Matches "\n\n<ANSWER>अ"
                    rf"is option\s*({escaped_char})",  # Matches "is option अ"
                    rf":\s*({escaped_char})",  # Matches ": अ"
                    rf"[Oo]ption\s*\(\s*({escaped_char})\s*",  # Matches "Option (अ)"
                    rf"\s({escaped_char})\.",  # Matches "अ." or "अ "
                    rf"({escaped_char})\.\s",
                    rf"\*\*({escaped_char})\*\*",
                    rf"\\boxed\{{({escaped_char})\}}",
                    rf":\s*({escaped_char})",
                    rf"\s*\(({escaped_char})\)",
                    rf"\n\nइस प्रकार, सही उत्तर विकल्प ({escaped_char}) है।",
                ]
            )

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            print(match)
            return match.group(1).upper()  # Return the matched choice in uppercase

    return None


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

    missing = get_missing_answers(data)
    print(f"Number of missing answers: {len(missing)}")

    def format_answers(data, pangea=False):
        formatted_data = []
        for sample in data:
            original = sample.get(prediction_field, None)
            answer = sample.get("reasoning", "")

            if original not in [0, 1, 2, 3]:
                if pangea:
                    answer = answer.split("Answer: assistant ")[-1]

                # Extract the choice from the response
                choice = extract_choice(answer)
                if choice:
                    try:
                        # If the choice is a non-Latin character, map it to Latin
                        for lang, mappings in NON_LATIN_TO_LATIN.items():
                            if choice in mappings:
                                # print(choice)
                                choice = mappings[
                                    choice
                                ]  # Map to Latin (e.g., "बी" -> "B")
                                break
                        # Convert the choice to a numeric value (0 for A, 1 for B, etc.)
                        sample[prediction_field] = ord(choice) - ord("A")
                    except Exception as e:
                        print(f"Error converting choice: {e}")

            formatted_data.append(sample)

        return formatted_data

    def format_answers_pangea(data, pangea=False):
        formatted_data = []
        for sample in data:
            original = sample[prediction_field]
            answer = sample["reasoning"]
            if original not in [0, 1, 2, 3]:
                if answer:
                    if pangea:
                        answer = answer.split("Answer: assistant ")[-1]
                    pattern = r"assistant <ANSWER>\s*([ABCD])\s*</ANSWER>"
                    match = re.search(pattern, answer)
                    if not match:
                        match = re.search(
                            r"assistant\s+([ABCD])", answer, re.IGNORECASE
                        )
                    if not match:
                        pattern = r"correct answer is <ANSWER>\s*([ABCD])\s*</ANSWER>"
                        match = re.search(pattern, answer)
                    if not match:
                        pattern = r"correct answer is\s*([ABCD])"
                        match = re.search(pattern, answer)
                    if not match:
                        pattern = r"correct answer is option\s*([ABCD])"
                        match = re.search(pattern, answer)
                    if not match:
                        match = re.search(r"assistant\s*<\s*([ABCD])\s*", answer)
                    if not match:
                        match = re.search(r"\n\n\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"\n\n<ANSWER>\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"is option\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"answer is:\s*([ABCD])", answer)
                    if not match:
                        match = re.search(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", answer)
                    if not match:
                        match = re.search(r"[Oo]ption\s*\(\s*([ABCD])\s*", answer)
                    if not match:
                        match = re.search(r"([ABCD])[\.\s]?", answer)
                    if match:
                        try:
                            letter = match.group(1).upper()
                            sample[prediction_field] = ord(letter) - ord("A")
                        except:
                            print(match)
                    if not match:
                        matches = re.findall(r"<ANSWER>\s*([ABCD])\s*</ANSWER>", answer)
                        if len(matches) >= 4:
                            letter = matches[3].upper()
                            sample[prediction_field] = ord(letter) - ord("A")

            formatted_data.append(sample)
        return formatted_data

    formatted_data = format_answers(data)
    assert len(formatted_data) == len(data)
    missing = get_missing_answers(formatted_data)
    print(f"Number of missing after answers formatting: {len(missing)}")
    import code

    code.interact(local=locals())
    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(formatted_data, f, indent=2)

        print(f"Formatted data saved to: {args.save_path}")


if __name__ == "__main__":
    main()
