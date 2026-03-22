"""ColBERT encoder — loads Reason-ModernColBERT and produces 128-dim token embeddings.

Uses the base model (transformers AutoModel) + the 768→128 projection head
from the model repo, avoiding the PyLate dependency entirely.

Usage as module:
    encoder = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    query_vecs = encoder.encode_query("Who won the Nobel Prize?")   # list[list[float]]
    doc_vecs = encoder.encode_docs(["doc1 text", "doc2 text"])      # list[list[list[float]]]
"""

import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class ColBERTEncoder:
    """Encode text into ColBERT 128-dim token embeddings.

    Inserts [Q] or [D] prefix tokens after [CLS], matching the training
    format of Reason-ModernColBERT (via PyLate's config_sentence_transformers.json).
    """

    def __init__(self, model_name: str = "lightonai/Reason-ModernColBERT", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # Register [Q] and [D] as special tokens (matching PyLate's training config)
        special_tokens = self.tokenizer.add_tokens(["[Q] ", "[D] "], special_tokens=True)
        if special_tokens > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.q_token_id = self.tokenizer.convert_tokens_to_ids("[Q] ")
        self.d_token_id = self.tokenizer.convert_tokens_to_ids("[D] ")

        # Load the 768→128 projection head
        proj_path = hf_hub_download(model_name, "1_Dense/model.safetensors")
        proj_weights = load_file(proj_path)
        self.projection = proj_weights["linear.weight"].to(device)  # (128, 768)

    def _insert_prefix_token(self, encoded: dict, token_id: int) -> dict:
        """Insert a prefix token at position 1 (after [CLS]) in the tokenized input."""
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        batch_size = input_ids.shape[0]

        # Create prefix column
        prefix = torch.full((batch_size, 1), token_id, dtype=input_ids.dtype, device=input_ids.device)
        ones = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)

        # Insert after position 0 ([CLS])
        new_input_ids = torch.cat([input_ids[:, :1], prefix, input_ids[:, 1:]], dim=1)
        new_attention_mask = torch.cat([attention_mask[:, :1], ones, attention_mask[:, 1:]], dim=1)

        result = {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_mask,
        }
        if "token_type_ids" in encoded:
            zeros = torch.zeros((batch_size, 1), dtype=encoded["token_type_ids"].dtype, device=input_ids.device)
            result["token_type_ids"] = torch.cat([encoded["token_type_ids"][:, :1], zeros, encoded["token_type_ids"][:, 1:]], dim=1)
        return result

    @torch.no_grad()
    def _encode(self, texts: list[str], max_length: int = 512, is_query: bool = False) -> list[torch.Tensor]:
        """Encode texts into L2-normalized 128-dim token embeddings.

        Returns a list of tensors, each (num_tokens, 128), one per input text.
        Padding tokens are stripped.
        """
        # Reserve 1 token for the [Q]/[D] prefix
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length - 1,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Insert [Q] or [D] prefix token after [CLS]
        prefix_id = self.q_token_id if is_query else self.d_token_id
        encoded = self._insert_prefix_token(encoded, prefix_id)

        hidden = self.model(**encoded).last_hidden_state  # (batch, seq, 768)
        projected = hidden @ self.projection.T  # (batch, seq, 128)
        projected = torch.nn.functional.normalize(projected, dim=-1)

        # Strip padding tokens using attention mask
        mask = encoded["attention_mask"]  # (batch, seq)
        results = []
        for i in range(projected.size(0)):
            length = mask[i].sum().item()
            results.append(projected[i, :length])  # (real_tokens, 128)

        return results

    def encode_query(self, text: str) -> list[list[float]]:
        """Encode a single query → list of 128-dim token vectors (with [Q] prefix)."""
        vecs = self._encode([text], max_length=128, is_query=True)
        return vecs[0].cpu().tolist()

    def encode_docs(self, texts: list[str], max_length: int = 512) -> list[list[list[float]]]:
        """Encode documents → list of (list of 128-dim token vectors) (with [D] prefix)."""
        vecs = self._encode(texts, max_length=max_length, is_query=False)
        return [v.cpu().tolist() for v in vecs]

    def encode_docs_tensors(self, texts: list[str], max_length: int = 512) -> list[torch.Tensor]:
        """Like encode_docs but returns GPU tensors (for batch index building)."""
        return self._encode(texts, max_length=max_length, is_query=False)
