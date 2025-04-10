from collections import Counter
from datasets import load_from_disk, concatenate_datasets, Dataset
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle
from PIL import Image


def main():
    random.seed(42)
    dataset = load_from_disk("dataset/full.hf")
    print(dataset)
    initial_questions = len(dataset)
    stats = {
        "multimodal_questions": 0,
        "questions_by_language": {},
        "multimodal_by_language": {},
        "questions_by_category": {},
        "multimodal_by_category": {},
        "text_by_category": {},
        "text_by_language": {},
        "text_by_language_and_category": {},
        "questions_by_language_and_category": {},
        "multimodal_by_language_and_category": {},
        "questions_by_general_category": {},
        "multimodal_by_general_category": {},
        "questions_by_language_and_general_category": {},
        "multimodal_by_language_and_general_category": {},
    }

    # check only >=4 options (cut to 4 afterwards)
    dataset = dataset.filter(lambda example: len(example["options"]) >= 4)
    only_four_or_more_options = len(dataset)

    # Count questions with exactly 4 options
    only_four_options = sum(1 for example in dataset if len(example["options"]) == 4)

    # filtering images in options
    dataset = dataset.filter(lambda example: ".png" not in example["options"][0])

    # filter questions where the image is missing
    def check_img(sample):
        if sample["image"] is not None:
            try:
                img = Image.open(sample["image"])
                return True
            except:
                return False
        else:
            return True

    dataset = dataset.filter(check_img)

    # Define category mappings
    normalized_category_mapping = {
        # STEM
        "Mathematics": {
            "Mathematical Reasoning",
            "Mathematical Abilities",
            "Numerical Ability",
            "Maths",
            "maths",
            "math",
            "Math",
            "Algebra",
            "Geometry",
            "Statistics",
            "Mathematics",
            "numerical ability",
            "mathematics",
        },
        "Chemistry": {"Chemistry", "chemistry"},
        "Biology": {
            "Biology",
            "Medicine/Biology",
            "Natural Science",
            "Life Sciences",
            "Biophysics",
            "Ecology and Evolution",
            "olympiads",
            "biology",
        },
        "Physics": {"Physics", "physics", "Science"},
        "Informatics": {
            "Computer Science",
            "computer science",
            "Computer Science and Information Technology",
            "Computer Knowledge",
            "computer concepts",
        },
        "Engineering": {
            "Electronics and Communications Engineering",
            "Mechanical Engineering",
            "Electrical Engineering",
            "Civil Engineering",
            "Aerospace Engineering",
            "Chemical Engineering",
            "Production and Industrial Engineering",
            "Textile Engineering and Fibre Science",
            "Biomedical Engineering",
            "Mining Engineering",
            "Geology Geophysics",
            "Geophysics",
            "Environmental Science & Engineering",
            "Engineering Sciences",
            "Electronics and Communication Engineering",
            "Metallurgical Engineering",
            "Naval Architecture and Marine Engineering",
            "Architecture and Planning",
            "Petroleum Engineering",
            "Biotechnology",
            "Agricultural Engineering",
            "Instrumentation Engineering",
            "electronics",
            "Geology",
            "Geomatics Engineering",
            "Amateur Radio",
        },
        # Health Sciences
        "Medicine": {
            "Medicine",
            "Medicine/Physiology",
            "Medicine/Health",
            "Nursing",
        },
        "Health": {"social_skills", "Health", "STEM"},
        "Language": {
            "Ukrainian Language and Literature",
            "Portuguese Language",
            "Portuguese Language, Literature",
            "Portuguese Language, Literature, History",
            "English",
            "English Language",
            "Reading_comprehension",
            "reading_comprehension",
            "Arabic Language",
        },
        # Humanities & Culture
        "Literature & Linguistics": {
            "Literature",
            "Humanities & Social Sciences - Linguistics",
        },
        "History": {
            "Indian History",
            "Ukrainian History",
            "History",
            "History and Civics",
        },
        "Culture": {"Culture", "Art History"},
        "Religion": {"Islamic Studies", "Christian Studies"},
        "Philosophy": {"Philosophy", "Philosophy and Logic"},
        # Social & Analytical Sciences
        "Geography": {"Geography"},
        "Social Sciences": {
            "Social Studies 2",
            "Civics",
            "Indian Politics",
            "Political Studies",
            "Demography",
            "Indian Constitution",
        },
        "Economics": {"Economics"},
        "Sociology": {"Sociology", "Psychology", "Psychology and Sociology"},
        "Reasoning": {
            "Reasoning",
            "General Intelligence and Reasoning",
            "Visual Reasoning",
            "Reasoning and General Intelligence",
            "Mental Ability",
            "mental ability",
            "IQ",
            "reasoning",
            "Aptitude",
        },
        "Design": {"drawing aptitude", "Design"},
        "Driving License": {
            "Driving License",
            "driving licence",
            "Driving_test",
            "Driver's License Exam",
            "Driving",
        },
        # General Knowledge, Culture and Arts
        "General Knowledge": {
            "general knowledge",
            "General Awareness",
            "Current Affairs",
        },
    }

    general_category_mapping = {
        "STEM": {
            "Mathematics",
            "Chemistry",
            "Biology",
            "Physics",
            "Informatics",
            "Engineering",
        },
        "Health Sciences": {"Medicine", "Health"},
        "Humanities & Culture": {
            "Language",
            "Literature & Linguistics",
            "History",
            "Philosophy",
            "Culture",
            "Religion",
        },
        "Social Sciences": {
            "Geography",
            "Social Sciences",
            "Economics",
            "Sociology",
        },
        "Reasoning": {"Reasoning"},
        "Practical Skills & General Knowledge": {
            "General Knowledge",
            "Driving License",
            "Design",
        },
    }

    def normalize_categories(example, category_mapping, general_category_mapping):
        original_category = example["category_en"]
        normalized_category = next(
            (
                norm_cat
                for norm_cat, raw_cats in category_mapping.items()
                if original_category in raw_cats
            ),
            "Other / Uncategorized",  # Default if no match
        )
        general_category = next(
            (
                gen_cat
                for gen_cat, norm_cats in general_category_mapping.items()
                if normalized_category in norm_cats
            ),
            "Other / Uncategorized",  # Default if no match
        )
        example["category_en"] = normalized_category
        example["general_category_en"] = general_category
        if example["general_category_en"] == "Other / Uncategorized":
            print(original_category)
        return example

    dataset = dataset.map(
        partial(
            normalize_categories,
            category_mapping=normalized_category_mapping,
            general_category_mapping=general_category_mapping,
        ),
        load_from_cache_file=False,
    )

    # Collect statistics
    def collect_stats(example, stats):
        lang = example["language"]
        category = example["category_en"]
        general_category = example["general_category_en"]
        is_multimodal = example["image"] is not None

        # Update multimodal count
        if is_multimodal:
            stats["multimodal_questions"] += 1

        # Update language statistics
        if lang not in stats["questions_by_language"]:
            stats["questions_by_language"][lang] = 0
            stats["multimodal_by_language"][lang] = 0
            stats["text_by_language"][lang] = 0
        stats["questions_by_language"][lang] += 1
        if is_multimodal:
            stats["multimodal_by_language"][lang] += 1
        else:
            stats["text_by_language"][lang] += 1

        # Update category statistics
        if category not in stats["questions_by_category"]:
            stats["questions_by_category"][category] = 0
            stats["multimodal_by_category"][category] = 0
            stats["text_by_category"][category] = 0
        stats["questions_by_category"][category] += 1
        if is_multimodal:
            stats["multimodal_by_category"][category] += 1
        else:
            stats["text_by_category"][category] += 1

        # Update general category statistics
        if general_category not in stats["questions_by_general_category"]:
            stats["questions_by_general_category"][general_category] = 0
            stats["multimodal_by_general_category"][general_category] = 0
        stats["questions_by_general_category"][general_category] += 1
        if is_multimodal:
            stats["multimodal_by_general_category"][general_category] += 1

        # Update language and category statistics
        key = (lang, category)
        if key not in stats["questions_by_language_and_category"]:
            stats["questions_by_language_and_category"][key] = 0
            stats["multimodal_by_language_and_category"][key] = 0
            stats["text_by_language_and_category"][key] = 0
        stats["questions_by_language_and_category"][key] += 1
        if is_multimodal:
            stats["multimodal_by_language_and_category"][key] += 1
        else:
            stats["text_by_language_and_category"][key] += 1

        key = (lang, general_category)
        if key not in stats["questions_by_language_and_general_category"]:
            stats["questions_by_language_and_general_category"][key] = 0
            stats["multimodal_by_language_and_general_category"][key] = 0
        stats["questions_by_language_and_general_category"][key] += 1
        if is_multimodal:
            stats["multimodal_by_language_and_general_category"][key] += 1
        return example

    dataset.map(partial(collect_stats, stats=stats), load_from_cache_file=False)

    # Print statistics
    print("NUM OPTIONS FILTER")
    print(f"{initial_questions=}")
    print(f"{only_four_or_more_options=}")
    print(f"{only_four_options=}")
    print("-" * 10)

    # minimal number of mm questions
    min_questions = 100
    mm_languages_to_keep = {
        lang
        for lang, count in stats["multimodal_by_language"].items()
        if count >= min_questions
    }
    text_languages_to_keep = {
        lang
        for lang, count in stats["text_by_language"].items()
        if count >= min_questions
    }

    def filter_examples(example):
        is_multimodal = example["image"] is not None
        if is_multimodal:
            return example["language"] in mm_languages_to_keep
        else:
            return example["language"] in text_languages_to_keep

    dataset = dataset.filter(filter_examples)

    print("\nMIN NUM QUESTIONS FILTER")
    original_num_languages = len(stats["questions_by_language"])

    filtered_num_languages = len(text_languages_to_keep.union(mm_languages_to_keep))
    print(f"Original number of languages: {original_num_languages}")
    print(f"Number of languages after filtering (>=100): {filtered_num_languages}")
    print(f"Number of questions after filtering (>=100): {len(dataset)}")

    # maximum number of multimodal questions per language to 1000
    multimodal_dataset = dataset.filter(lambda example: example["image"] is not None)
    text_dataset = dataset.filter(lambda example: example["image"] is None)

    assert len(dataset) == len(multimodal_dataset) + len(text_dataset)

    def cap_mm_questions_per_language(dataset, stats, max_questions_per_language=1000):
        capped_dataset = []

        # Iterate over each language
        for lang, lang_count in tqdm(stats["multimodal_by_language"].items()):
            if lang_count <= max_questions_per_language:
                # If the language has fewer than max_questions_per_language, keep all questions
                capped_dataset.extend([ex for ex in dataset if ex["language"] == lang])
            else:
                categories_in_lang = {
                    cat: count
                    for (l, cat), count in stats[
                        "multimodal_by_language_and_category"
                    ].items()
                    if l == lang
                }
                # Calculate the target number of questions per category
                total_categories = sum(
                    1 for value in categories_in_lang.values() if value != 0
                )
                target_per_category = max_questions_per_language // total_categories
                sorted_categories = dict(
                    sorted(categories_in_lang.items(), key=lambda item: item[1])
                )
                num_questions = list(sorted_categories.values())
                added = 0
                extra = 0

                for t, (category, count) in enumerate(sorted_categories.items()):
                    if count == 0:
                        continue
                    category_questions = [
                        ex
                        for ex in dataset
                        if ex["language"] == lang and ex["category_en"] == category
                    ]
                    total_categories -= 1
                    if count < target_per_category:
                        capped_dataset.extend(category_questions)
                        extra += target_per_category - count
                        added += count
                    else:
                        multimodal_questions = [
                            ex for ex in category_questions if ex["image"] is not None
                        ]

                        cap = min(
                            target_per_category + (extra // (total_categories + 1)),
                            max_questions_per_language - added,
                        )
                        extra = len(multimodal_questions) - target_per_category
                        if len(multimodal_questions) < cap:
                            capped_dataset.extend(multimodal_questions)
                            added += len(multimodal_questions)
                        else:
                            sampled_questions = random.sample(
                                multimodal_questions,
                                cap,
                            )
                            capped_dataset.extend(sampled_questions)
                            added += len(sampled_questions)

        return capped_dataset

    multimodal_dataset = cap_mm_questions_per_language(multimodal_dataset, stats)
    multimodal_dataset = Dataset.from_list(multimodal_dataset)

    def cap_text_questions_per_language(
        dataset, stats, max_questions_per_language=1000
    ):
        capped_dataset = []

        # Iterate over each language
        for lang, lang_count in tqdm(stats["text_by_language"].items()):
            if lang_count <= max_questions_per_language:
                # If the language has fewer than max_questions_per_language, keep all questions
                capped_dataset.extend([ex for ex in dataset if ex["language"] == lang])
            else:
                categories_in_lang = {
                    cat: count
                    for (l, cat), count in stats[
                        "text_by_language_and_category"
                    ].items()
                    if l == lang
                }
                # Calculate the target number of questions per category
                total_categories = sum(
                    1 for value in categories_in_lang.values() if value != 0
                )
                target_per_category = max_questions_per_language // total_categories
                sorted_categories = dict(
                    sorted(categories_in_lang.items(), key=lambda item: item[1])
                )
                num_questions = list(sorted_categories.values())
                added = 0
                extra = 0

                for t, (category, count) in enumerate(sorted_categories.items()):
                    if count == 0:
                        continue
                    category_questions = [
                        ex
                        for ex in dataset
                        if ex["language"] == lang and ex["category_en"] == category
                    ]
                    total_categories -= 1
                    if count < target_per_category:
                        capped_dataset.extend(category_questions)
                        extra += target_per_category - count
                        added += count
                    else:
                        text_questions = [
                            ex for ex in category_questions if ex["image"] is None
                        ]

                        cap = min(
                            target_per_category + (extra // (total_categories + 1)),
                            max_questions_per_language - added,
                        )
                        extra = len(text_questions) - target_per_category
                        if len(text_questions) < cap:
                            capped_dataset.extend(text_questions)
                            added += len(text_questions)
                        else:
                            sampled_questions = random.sample(
                                text_questions,
                                cap,
                            )
                            capped_dataset.extend(sampled_questions)
                            added += len(sampled_questions)

        return capped_dataset

    text_dataset = cap_text_questions_per_language(text_dataset, stats)
    text_dataset = Dataset.from_list(text_dataset)

    mm_language_counts = Counter(multimodal_dataset["language"])
    filtered_splits = {}

    def filter_and_sample_text(dataset, mm_counts):
        # Filter only the samples that match the current language
        dataset = dataset.filter(lambda example: example["language"] in mm_counts)

        language_to_samples = {}
        for example in dataset:
            lang = example["language"]
            if lang not in language_to_samples:
                language_to_samples[lang] = []
            language_to_samples[lang].append(example)

        sampled_examples = []
        for lang, samples in language_to_samples.items():
            max_samples = mm_language_counts[lang]
            if len(samples) > max_samples:
                sampled_examples.extend(random.sample(samples, max_samples))
            else:
                sampled_examples.extend(samples)

        # Convert the sampled examples back to a Hugging Face Dataset
        sampled_dataset = Dataset.from_list(sampled_examples)
        return sampled_dataset

    # Combine all filtered language datasets into a single dataset
    text_dataset = filter_and_sample_text(text_dataset, mm_language_counts)

    # compute new statistics
    stratified_stats = {
        "multimodal_questions": 0,
        "questions_by_language": {},
        "multimodal_by_language": {},
        "questions_by_category": {},
        "multimodal_by_category": {},
        "text_by_category": {},
        "text_by_language": {},
        "text_by_language_and_category": {},
        "questions_by_language_and_category": {},
        "multimodal_by_language_and_category": {},
        "questions_by_general_category": {},
        "multimodal_by_general_category": {},
        "questions_by_language_and_general_category": {},
        "multimodal_by_language_and_general_category": {},
    }

    stratified_dataset = concatenate_datasets([multimodal_dataset, text_dataset])

    stratified_dataset.map(
        partial(collect_stats, stats=stratified_stats), load_from_cache_file=False
    )

    def correct_question(sample):
        if (
            sample["question"]
            == "Cinq cartes numérotées de 1 à 5 sont alignées. А chaque tour, on échange 2 cartes. Combien de tours faut-il, au minimum, pour arriver à la ligne du dessous ?"
        ):
            if (
                sample["options"][0]
                == "celui d'un poids de 4305 kg et d'une largeur de 315 cm"
            ):
                sample["question"] = (
                    "Deux panneaux de signalisation routière sont placés à l'entrée du pont de mon village. L'un marque le poids maximum autorisé et l'autre la largeur maximum autorisée. Lequel des ces camions a le droit de traverser le pont?"
                )
        return sample

    def reduce_options_to_four(example):
        if len(example["options"]) == 4:
            return example
        options = example["options"]
        answer_index = example["answer"]

        correct_option = options[answer_index]
        wrong_options = [
            option for i, option in enumerate(options) if i != answer_index
        ]
        wrong_options = random.sample(wrong_options, 3)
        new_options = wrong_options + [correct_option]
        random.shuffle(new_options)
        new_answer_index = new_options.index(correct_option)
        example["options"] = new_options
        example["answer"] = new_answer_index
        return example

    stratified_dataset = stratified_dataset.map(
        correct_question, load_from_cache_file=False
    )

    with open("dataset/stast.pkl", "wb") as f:
        pickle.dump(stats, f)

    with open("dataset/stratified_stats.pkl", "wb") as f:

        pickle.dump(stratified_stats, f)

    stratified_dataset.save_to_disk("dataset/stratified_dataset_.hf")


if __name__ == "__main__":
    main()
