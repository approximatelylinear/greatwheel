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
    """Encode text into ColBERT 128-dim token embeddings."""

    def __init__(self, model_name: str = "lightonai/Reason-ModernColBERT", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

        # Load the 768→128 projection head
        proj_path = hf_hub_download(model_name, "1_Dense/model.safetensors")
        proj_weights = load_file(proj_path)
        self.projection = proj_weights["linear.weight"].to(device)  # (128, 768)

    @torch.no_grad()
    def _encode(self, texts: list[str], max_length: int = 512) -> list[torch.Tensor]:
        """Encode texts into L2-normalized 128-dim token embeddings.

        Returns a list of tensors, each (num_tokens, 128), one per input text.
        Padding tokens are stripped.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

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
        """Encode a single query → list of 128-dim token vectors."""
        vecs = self._encode([text], max_length=128)
        return vecs[0].cpu().tolist()

    def encode_docs(self, texts: list[str], max_length: int = 512) -> list[list[list[float]]]:
        """Encode documents → list of (list of 128-dim token vectors), one per doc."""
        vecs = self._encode(texts, max_length=max_length)
        return [v.cpu().tolist() for v in vecs]

    def encode_docs_tensors(self, texts: list[str], max_length: int = 512) -> list[torch.Tensor]:
        """Like encode_docs but returns GPU tensors (for batch index building)."""
        return self._encode(texts, max_length=max_length)
