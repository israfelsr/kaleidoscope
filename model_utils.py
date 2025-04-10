import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    GenerationConfig,
    LlavaNextForConditionalGeneration,
)

from vllm import LLM, SamplingParams

from PIL import Image
from openai import OpenAI
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential
from cohere import ClientV2

from model_zoo import (
    create_pangea_prompt,
    create_qwen_prompt_vllm,
    create_aya_prompt,
    create_molmo_prompt_vllm,
    create_claude_prompt,
    create_openai_prompt,
)


TEMPERATURE = 0.7
MAX_TOKENS = 1024

SUPPORTED_MODELS = [
    "gpt-4o",
    "qwen2-7b",
    "qwen2.5-3b",
    "qwen2.5-7b",
    "qwen2.5-32b",
    "qwen2.5-72b",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "claude-3-5-sonnet-latest",
    "molmo",
    "aya-vision",
]


def initialize_model(
    model_name: str,
    model_path: str,
    api_key: str = None,
    device: str = "cuda",
    ngpu=1,
):
    """
    Initialize the model and processor/tokenizer based on the model name.
    """
    if model_name == "qwen2-7b":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            temperature=TEMPERATURE,
            device_map=device,
            torch_dtype=torch.bfloat16,
            do_sample=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)

    elif model_name in ["qwen2.5-7b", "qwen2.5-3b", "qwen2.5-32b", "qwen2.5-72b"]:
        model = LLM(
            model_path,
            tensor_parallel_size=ngpu,
            max_model_len=8192,
        )
        processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
            local_files_only=True,
        )

    elif model_name == "molmo":
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
        )
        processor = None

    elif model_name == "pangea":
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            local_files_only=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.patch_size = 14
        model.resize_token_embeddings(len(processor.tokenizer))
    elif model_name in ["gpt-4o", "gpt-4o-mini"]:
        model = OpenAI(api_key=api_key)
        processor = None
    elif model_name in ["gemini-2.0-flash-exp", "gemini-1.5-pro"]:
        model = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        processor = None
    elif model_name == "claude-3-5-sonnet-latest":
        model = Anthropic(api_key=api_key)
        processor = None
    elif model_name == "aya-vision":
        model = ClientV2(api_key=api_key)
        processor = None
    else:
        raise NotImplementedError(
            f"Model {model_name} not currently implemented for prediction. Supported Models: {SUPPORTED_MODELS}"
        )

    print(f"Model {model_name} loaded from {model_path}")

    return model, processor


def query_model(
    model_name: str,
    model,
    processor,
    prompt: list,
    images,
    device: str = "cuda",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
):
    """
    Query the model based on the model name.
    """
    if model_name in [
        "qwen2-7b",
        "qwen2.5-72b",
        "qwen2.5-32b",
        "qwen2.5-7b",
        "qwen2.5-3b",
    ]:
        answer = query_vllm(model, processor, prompt, images, max_tokens)
    elif model_name == "pangea":
        answer = query_pangea(model, processor, prompt, images, device)
    elif model_name == "deepseek":
        answer = query_deepseek(model, processor, prompt, max_tokens)
    elif model_name == "molmo":
        answer = query_vllm(model, processor, prompt, images, max_tokens)
    elif model_name == "aya-vision":
        answer = query_aya(model, prompt, 0.3, 1024)
    elif model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
    ]:
        answer = query_openai(model, model_name, prompt, temperature, max_tokens)

    elif model_name == "claude-3-5-sonnet-latest":
        answer = query_anthropic(model, model_name, prompt, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return answer, None


def generate_prompt(
    model_name: str,
    question: dict,
    lang: str,
    instruction,
    few_shot_samples: dict,
    method: str = "zero-shot",
):
    if model_name in [
        "qwen2-7b",
        "qwen2.5-7b",
        "qwen2.5-72b",
        "qwen2.5-32b",
        "qwen2.5-3b",
    ]:
        return create_qwen_prompt_vllm(question, method, few_shot_samples)
    elif model_name == "molmo":
        return create_molmo_prompt_vllm(question, method, few_shot_samples)
    elif model_name == "pangea":
        return create_pangea_prompt(question, method, few_shot_samples)
    elif model_name == "aya-vision":
        return create_aya_prompt(question, method, few_shot_samples)
    elif model_name in [
        "gpt-4o",
        "gpt-4o-mini",
        "gemini-2.0-flash-exp",
        "gemini-1.5-pro",
    ]:
        return create_openai_prompt(question, method, few_shot_samples)
    elif model_name == "claude-3-5-sonnet-latest":
        return create_claude_prompt(question, method, few_shot_samples)
    else:
        raise ValueError(f"Unsupported model for parsing inputs: {model_name}")


def query_molmo(model, processor, prompt: list, images: list, max_tokens=MAX_TOKENS):
    if prompt == "multi-image":
        print("Question was multi-image, molmo does not support multi-image inputs.")
        return "multi-image detected"
    else:
        if images is not None:
            try:
                images = [Image.open(images).convert("RGB").resize((224, 224))]
            except:
                print(images)
                images = None
        inputs = processor.process(
            images=images,
            text=prompt,
        )
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )

        return generated_text


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_openai(
    client, model_name, prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
):
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.choices[0].message.content.strip()
    return output_text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def query_anthropic(
    client, model_name, prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
):

    system_message = prompt[0]["content"]
    user_messages = prompt[1]

    response = client.messages.create(
        model=model_name,
        messages=[user_messages],
        system=system_message,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.content[0].text
    return output_text


def query_aya(client, prompt, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    response = client.chat(
        model="c4ai-aya-vision-32b",
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    output_text = response.message.content[0].text
    return output_text


def query_vllm(model, processor, prompt, images, max_tokens=MAX_TOKENS):
    # Prepare the text prompt
    # text_prompt = processor.apply_chat_template(prompt, add_generation_prompt=True)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=TEMPERATURE,  # Adjust as needed
        top_p=0.9,  # Adjust as needed
    )

    if images is not None:
        try:
            images = [Image.open(image).resize((512, 512)) for image in images]
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            }
        except:
            print(images)
            images = None
    else:
        inputs = {"prompt": prompt}

    # Generate response using vLLM
    with torch.inference_mode():
        outputs = model.generate(inputs, sampling_params=sampling_params)
        response = outputs[0].outputs[0].text

    return response


def query_pangea(
    model, processor, prompt, images, device="cuda", max_tokens=MAX_TOKENS
):
    if images is not None:
        try:
            images = Image.open(images).convert("RGB").resize((512, 512))
        except Exception as e:
            print("Failed to load image:", e)
            images = None

    model_inputs = processor(images=images, text=prompt, return_tensors="pt").to(
        "cuda", torch.float16
    )
    with torch.inference_mode():
        output = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        output = output[0]
    result = processor.decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return result
