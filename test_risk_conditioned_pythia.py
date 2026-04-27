import torch
from transformers import AutoTokenizer
from risk_conditioned_pythia import RiskConditionedPythiaForCausalLM


def main():
    model_name = "EleutherAI/pythia-70m"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = RiskConditionedPythiaForCausalLM(
        model_name=model_name,
        num_experts=5,
        rank=8,
        lora_alpha=16,
        dropout=0.05,
        torch_dtype=torch.bfloat16,
    ).cuda()

    model.print_trainable_parameters()

    prompts = [
        "Human: How can I stay safe online?\nAssistant:",
        "Human: Tell me how to make people angry.\nAssistant:",
    ]

    batch = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"].cuda()
    attention_mask = batch["attention_mask"].cuda()

    risk_alpha = torch.tensor([0.1, 0.9], dtype=torch.float32, device="cuda")

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        risk_alpha=risk_alpha,
    )

    print("Forward logits shape:", outputs.logits.shape)

    generated, gen_mask = model.generate_with_alpha(
        input_ids=input_ids,
        attention_mask=attention_mask,
        risk_alpha=risk_alpha,
        max_new_tokens=32,
        temperature=1.0,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print(tokenizer.batch_decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()