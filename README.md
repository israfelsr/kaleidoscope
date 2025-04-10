# Kaleidoscope: In-language Exams for Massively Multilingual Vision Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2504.07072-b31b1b.svg)](https://arxiv.org/abs/2504.07072)
[![HuggingFace](https://img.shields.io/badge/🤗%20Datasets-Kaleidoscope-yellow)](https://hf.co/datasets/CohereForAI/kaleidoscope)

Official repository for *Kaleidoscope*, the **a comprehensive multilingual multimodal exam benchmark** evaluating VLMs across:
- **18 languages** (Bengali → Spanish)
- **14 subjects** (STEM to Humanities) 
- **20,911 questions** (55% requiring image understanding)

## Downloading the Dataset
```python
from datasets import load_dataset
dataset = load_dataset(CohereForAI/kaleidoscope)
```

## Running Inference
```python
python main.py \
--model <model_name> \
--dataset <dataset_name_or_path> \
--model_path <model_path> \
--api_key <api-if-needed>
```