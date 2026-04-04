import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYS_PROMPT = """
You are a careful math reasoning assistant.
Solve the problem step by step, keep the reasoning internally consistent,
and give the final answer clearly.
If the prompt contains supporting context, use it only when it is relevant.
"""


def get_data():
    with open("./data/raw/two_hop_math_reasoning.json", "r") as f:
        data = json.load(f)
    return data

def generate_with_logit_cached(ids, model, tokenizer, max_tokens=200, temperature=0):
    generated_tokens = []
    out = {}

    eos_token_id = tokenizer.eos_token_id
    past_key_values = None
    current_ids = ids

    with torch.inference_mode():
        for idx, _ in enumerate(range(max_tokens)):
            outputs = model.forward(
                current_ids,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values

            # Cache only the next-token distribution instead of full-sequence logits.
            next_token_logits = outputs.logits[:, -1, :]
            out[idx] = next_token_logits.detach().cpu()

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1)
            else:
                next_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token = next_ids.item()
            if token == eos_token_id:
                break

            generated_tokens.append(token)
            current_ids = next_ids

    print(tokenizer.decode(generated_tokens))
    return generated_tokens, out

def generate_with_forward_cached(ids, model, tokenizer, max_tokens=200, temperature=0):
    generated_tokens = []
    out = {}
    
    for idx, _ in enumerate(range(max_tokens)):
        outputs = forward_with_token_activations(model, ids)  # (B, T, vocab_size)
        out[idx] = outputs["activation"]
        logits = outputs["logits"]
        logits = logits[:, -1, :]  # (B, vocab_size)
        if temperature > 0:
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1)
        else:
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat((ids, next_ids), dim=1)
        token = next_ids.item()
        if token == tokenizer.encode(tokenizer.eos_token)[0]:
            break
        generated_tokens.append(token)

    print(tokenizer.decode(generated_tokens))
    return generated_tokens, out


def forward_with_token_activations(model, input_ids):
    output = model.forward(input_ids, output_hidden_states=True)
    out = {}
    all_hidden_states = []
    for idx, hidden_state in enumerate(output.hidden_states):
        hidden_state_cpu = hidden_state.detach().cpu()
        layer_name = f"layer_{idx}"
        out[layer_name] = hidden_state_cpu
        all_hidden_states.append(hidden_state_cpu)

    mean_pooled_across_layers = torch.stack(all_hidden_states, dim=0).mean(dim=0)
    out["mean_pooled_across_layers"] = mean_pooled_across_layers
    out["mean_pooled_across_layers_and_sequence"] = mean_pooled_across_layers.mean(
        dim=1
    )

    return {
        "activation": out,
        "logits": output.logits,
    }


def get_activations_across_all_tokens(
    model,
    tokenizer,
    data_point,
    output_path: str | Path | None = None,
):
    output = {'data_point': data_point}

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    
    answer, act = generate_with_forward_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages

    output["response_0"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }
    str_answer = tokenizer.decode(answer)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
        {"role": "assistant", "content": str_answer},
        {"role": "user", "content": data_point.get("hop_1")['description']},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    answer, act = generate_with_forward_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages
    
    output["response_1"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }
    str_answer_1 = tokenizer.decode(answer)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
        {"role": "assistant", "content": str_answer},
        {"role": "user", "content": data_point.get("hop_1")['description']},
        {"role": "assistant", "content": str_answer_1},  
        {"role": "user", "content": data_point.get("hop_2")['description']},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    answer, act = generate_with_forward_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages
    
    output["response_2"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }

    if output_path is None:
        stem = data_point.get("id")
        output_path = Path("results") / f"{stem}_math_reasoning.pt"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output_path

def get_logits_across_all_tokens(
    model,
    tokenizer,
    data_point,
    output_path: str | Path | None = None,
):
    output = {'data_point': data_point}

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    
    answer, act = generate_with_logit_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages

    output["response_0"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }
    str_answer = tokenizer.decode(answer)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
        {"role": "assistant", "content": str_answer},
        {"role": "user", "content": data_point.get("hop_1")['description']},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )

    answer, act = generate_with_logit_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages
    
    output["response_1"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }
    str_answer_1 = tokenizer.decode(answer)

    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": data_point.get("question")},
        {"role": "assistant", "content": str_answer},
        {"role": "user", "content": data_point.get("hop_1")['description']},
        {"role": "assistant", "content": str_answer_1},  
        {"role": "user", "content": data_point.get("hop_2")['description']},
    ]
    tokenized_messages = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    answer, act = generate_with_logit_cached(
        tokenized_messages.to(model.device), model=model, tokenizer=tokenizer
    )
    del tokenized_messages
    
    output["response_2"] = {
        "answer_token_ids": answer,
        "answer_text": tokenizer.decode(answer),
        "activations": act,
    }

    if output_path is None:
        stem = data_point.get("id")
        output_path = Path("results") / f"{stem}_math_reasoning.pt"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)
    return output_path

if __name__ == "__main__":
    #check
    # messages = [
    #     {"role": "system", "content": 'You are a normal assistant'},
    #     {"role": "user", "content": 'Hello, how are you?'},
    # ]
    data = get_data()
    model = AutoModelForCausalLM.from_pretrained('unsloth/Qwen2.5-3B-Instruct', torch_dtype=torch.bfloat16).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained('unsloth/Qwen2.5-3B-Instruct')

    data_points = data['questions']
    get_logits_across_all_tokens(model, tokenizer, data_points[0])
    # get_activations_across_all_tokens(model, tokenizer, data_points[0])
    # tokenized_messages = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=True,
    #     return_tensors='pt',
    #     add_generation_prompt=True,
    # )
    # # print(tokenized_messages.shape)
    # # out = model.forward(tokenized_messages.to('cuda'), output_hidden_states=True)
    # # print(out.logits.shape)
    # # print(out.hidden_states[0].shape)
    # max_tokens = 200
    # generated_tokens = []
    # temperature = 0
    # ids = tokenized_messages
    # # ids = torch.tensor([tokens], dtype=torch.long, device='cuda') # add batch dim
    # for idx, _ in enumerate(range(max_tokens)):
    #     outputs = model.forward(ids.to('cuda'), output_hidden_states=True)# (B, T, vocab_size)
    #     logits = outputs.logits
    #     logits = logits[:, -1, :] # (B, vocab_size)
    #     if temperature > 0:
    #         logits = logits / temperature
    #         probs = torch.nn.functional.softmax(logits, dim=-1)
    #         next_ids = torch.multinomial(probs, num_samples=1, generator=None)
    #     else:
    #         next_ids = torch.argmax(logits, dim=-1, keepdim=True)
    #     ids = torch.cat((ids.to('cuda'), next_ids.to('cuda')), dim=1)
    #     token = next_ids.item()
    #     if token == tokenizer.encode(tokenizer.eos_token)[0]:
    #         print(f'Stopped: {len(generated_tokens)}')
    #         break
    #     generated_tokens.append(token)
    
    # print(tokenizer.decode(generated_tokens))

    pass
