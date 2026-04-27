import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


class RiskLoRALinear(nn.Module):
    """
    Alpha-conditioned LoRA wrapper for nn.Linear.

    y = W_0 x + scale * sum_k m_k(alpha) B_k A_k x
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int = 5,
        rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.num_experts = num_experts
        self.rank = rank
        self.scaling = lora_alpha / rank
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.Parameter(
            torch.zeros(num_experts, rank, self.in_features)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(num_experts, self.out_features, rank)
        )

        self.gate = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, num_experts),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, risk_alpha: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden]
        risk_alpha: [batch] or [batch, 1]
        """

        base_out = self.base(x)

        if risk_alpha is None:
            raise ValueError("RiskLoRALinear requires risk_alpha.")

        if risk_alpha.dim() == 1:
            risk_alpha = risk_alpha[:, None]

        risk_alpha = risk_alpha.to(device=x.device, dtype=x.dtype)

        mix = torch.softmax(self.gate(risk_alpha), dim=-1)  # [B, K]

        x_drop = self.dropout(x)

        expert_outputs = []
        for k in range(self.num_experts):
            hidden = F.linear(x_drop, self.lora_A[k])   # [B, S, r]
            out = F.linear(hidden, self.lora_B[k])      # [B, S, H]
            expert_outputs.append(out)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, K, S, H]
        mix = mix[:, :, None, None]                          # [B, K, 1, 1]

        lora_out = (mix * expert_outputs).sum(dim=1)

        return base_out + self.scaling * lora_out


def patch_pythia_attention_lora(
    model: nn.Module,
    target_names: Tuple[str, ...] = ("query_key_value", "dense"),
    num_experts: int = 5,
    rank: int = 8,
    lora_alpha: int = 16,
    dropout: float = 0.05,
) -> nn.Module:
    """
    Patch Pythia/GPT-NeoX attention Linear layers.

    For Pythia, the main attention modules are usually:
        attention.query_key_value
        attention.dense

    We only patch children whose names match target_names.
    """

    replaced = []

    for module_name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if child_name in target_names and isinstance(child, nn.Linear):
                wrapped = RiskLoRALinear(
                    base_linear=child,
                    num_experts=num_experts,
                    rank=rank,
                    lora_alpha=lora_alpha,
                    dropout=dropout,
                )
                setattr(module, child_name, wrapped)
                replaced.append(f"{module_name}.{child_name}")

    if len(replaced) == 0:
        raise RuntimeError(
            "No attention Linear layers were patched. "
            "Please check target_names for this Pythia version."
        )

    print(f"Patched {len(replaced)} Linear layers with RiskLoRA:")
    for name in replaced[:10]:
        print(f"  - {name}")
    if len(replaced) > 10:
        print(f"  ... and {len(replaced) - 10} more")

    return model


class RiskConditionedPythiaForCausalLM(nn.Module):
    """
    Pythia causal LM with alpha-conditioned attention LoRA.

    Usage:
        model = RiskConditionedPythiaForCausalLM("EleutherAI/pythia-70m")
        out = model(input_ids, attention_mask, risk_alpha)
    """

    def __init__(
        self,
        model_name: str = "EleutherAI/pythia-70m",
        num_experts: int = 5,
        rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.05,
        torch_dtype=torch.bfloat16,
        device_map: Optional[str] = None,
    ):
        super().__init__()

        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        self.model.config.use_cache = False
        self.model.config.output_hidden_states = True

        self.model = patch_pythia_attention_lora(
            self.model,
            target_names=("query_key_value", "dense"),
            num_experts=num_experts,
            rank=rank,
            lora_alpha=lora_alpha,
            dropout=dropout,
        )

        self.freeze_base_model()

    def freeze_base_model(self):
        """
        Only train RiskLoRA parameters and gate network.
        """

        for name, param in self.model.named_parameters():
            if (
                "lora_A" in name
                or "lora_B" in name
                or "gate" in name
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def print_trainable_parameters(self):
        trainable = 0
        total = 0

        for _, p in self.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        pct = 100 * trainable / total

        print(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({pct:.4f}%)"
        )

    def _call_with_risk_alpha(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        risk_alpha: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        """
        Temporarily inject risk_alpha into every RiskLoRALinear layer.

        This avoids rewriting the HuggingFace GPT-NeoX forward code.
        """

        old_forwards = []

        for module in self.model.modules():
            if isinstance(module, RiskLoRALinear):
                old_forward = module.forward

                def make_forward(m):
                    def forward_with_alpha(x):
                        return RiskLoRALinear.forward(
                            m,
                            x,
                            risk_alpha=risk_alpha,
                        )
                    return forward_with_alpha

                module.forward = make_forward(module)
                old_forwards.append((module, old_forward))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )

        for module, old_forward in old_forwards:
            module.forward = old_forward

        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        risk_alpha: torch.Tensor,
        output_hidden_states: bool = True,
    ):
        return self._call_with_risk_alpha(
            input_ids=input_ids,
            attention_mask=attention_mask,
            risk_alpha=risk_alpha,
            output_hidden_states=output_hidden_states,
        )

    @torch.no_grad()
    def generate_with_alpha(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        risk_alpha: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        """
        Simple generation loop that always passes risk_alpha.

        This is slower than HF generate(), but easier and reliable for the first version.
        """

        self.eval()

        generated = input_ids
        cur_mask = attention_mask

        finished = torch.zeros(
            input_ids.size(0),
            dtype=torch.bool,
            device=input_ids.device,
        )

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated,
                attention_mask=cur_mask,
                risk_alpha=risk_alpha,
                output_hidden_states=False,
            )

            logits = outputs.logits[:, -1, :]

            if temperature is not None and temperature > 0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            if top_p is not None and top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)

                keep = cum_probs <= top_p
                keep[:, 0] = True

                filtered_probs = torch.zeros_like(probs)
                filtered_probs.scatter_(
                    dim=-1,
                    index=sorted_idx,
                    src=sorted_probs * keep,
                )

                probs = filtered_probs / filtered_probs.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)

            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished[:, None],
                    torch.full_like(next_token, pad_token_id or eos_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=-1)
            cur_mask = torch.cat([cur_mask, torch.ones_like(next_token)], dim=-1)

            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                if finished.all():
                    break

        return generated, cur_mask

    def save_risk_lora(self, path: str):
        """
        Save only trainable risk-conditioned LoRA parameters.
        """

        state = {
            k: v.cpu()
            for k, v in self.state_dict().items()
            if "lora_A" in k or "lora_B" in k or "gate" in k
        }

        torch.save(
            {
                "model_name": self.model_name,
                "risk_lora_state": state,
            },
            path,
        )

    def load_risk_lora(self, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt["risk_lora_state"]

        current = self.state_dict()
        current.update(state)
        self.load_state_dict(current)