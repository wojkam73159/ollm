from ollm import Inference, file_get_contents, TextStreamer


def main():
    # Initialize inference with desired model and device
    o = Inference("llama3-1B-chat", device="cuda:0", logging=True)  # llama3-1B/3B/8B-chat, gpt-oss-20B, qwen3-next-80B
    o.ini_model(models_dir="./models/", force_download=False)

    # (optional) offload some layers to CPU for speed boost on limited VRAM
    o.offload_layers_to_cpu(layers_num=2)

    # Optional: enable disk-backed KV cache for longer contexts
    past_key_values = o.DiskCache(cache_dir="./kv_cache/")  # set None if context is small

    # Stream tokens to stdout as they are generated
    text_streamer = TextStreamer(o.tokenizer, skip_prompt=True, skip_special_tokens=False)

    messages = [
        {"role": "system", "content": "You are helpful AI assistant"},
        {"role": "user", "content": "List planets"},
    ]

    input_ids = o.tokenizer.apply_chat_template(
        messages,
        reasoning_effort="minimal",
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(o.device)

    outputs = o.model.generate(
        input_ids=input_ids,
        past_key_values=past_key_values,
        max_new_tokens=500,
        streamer=text_streamer,
    ).cpu()

    answer = o.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=False)
    print("\n\nFinal answer:\n" + answer)


if __name__ == "__main__":
    main()

