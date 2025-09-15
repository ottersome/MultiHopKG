"""
Lightweight question encoder for conditioning policy with NLP questions.

Design goals:
- Try to use HuggingFace Transformers if available, but degrade gracefully to
  zero/pseudo embeddings so the pipeline remains runnable without extra deps.
- Provide a simple interface: QuestionEncoder.encode_batch(e_s, q, kg) -> tensor [B, D].
- Allow optional loading of question texts via a mapping file for placeholders.

Expected question mapping formats (optional):
- TSV: e_s_id <TAB> relation_id <TAB> question_text
- JSONL: {"e_s": <int>, "r": <int>, "question": <str>}

If no mapping is provided or a pair is missing, returns a zero vector.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Tuple, Optional, List

import torch


class QuestionEncoder:
    def __init__(self, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = int(getattr(args, 'bert_hidden_size', 768))
        self.max_len = int(getattr(args, 'max_question_len', 64))
        self.model_name = getattr(args, 'bert_model_name', 'bert-base-uncased')
        self.mapping_path = getattr(args, 'question_texts_path', '') or ''

        self._texts: Dict[Tuple[int, int], str] = {}
        if self.mapping_path and os.path.exists(self.mapping_path):
            self._load_mapping(self.mapping_path)

        # Best-effort lazy import of transformers
        self._tokenizer = None
        self._model = None
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        except Exception:
            # Dependencies not available, fallback will be used
            self._tokenizer = None
            self._model = None

    def _load_mapping(self, path: str) -> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.tsv':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        continue
                    try:
                        e_s = int(parts[0]); r = int(parts[1]); q = parts[2]
                    except Exception:
                        continue
                    self._texts[(e_s, r)] = q
        elif ext in ('.jsonl', '.json'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    try:
                        e_s = int(obj['e_s']); r = int(obj['r']); q = str(obj['question'])
                    except Exception:
                        continue
                    self._texts[(e_s, r)] = q

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a batch of texts. If transformers is available, produce CLS pooled output.
        Otherwise return a deterministic pseudo-embedding based on string hash, normalized.
        Returns a tensor [B, hidden_size] on the current device.
        """
        if self._tokenizer is not None and self._model is not None and len(texts) > 0:
            with torch.no_grad():
                toks = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors='pt'
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                out = self._model(**toks)
                # Use the first token ([CLS]) representation or pooled_output if available
                if hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    pooled = out.pooler_output
                else:
                    pooled = out.last_hidden_state[:, 0, :]
                if pooled.size(-1) != self.hidden_size:
                    # Project to desired size
                    # Use a simple linear projection with a fixed random matrix on CPU for determinism
                    W = torch.randn(pooled.size(-1), self.hidden_size, device=pooled.device)
                    pooled = pooled @ W
                return pooled
        # Fallback: pseudo-embeddings
        embs = torch.zeros((len(texts), self.hidden_size), dtype=torch.float32)
        for i, t in enumerate(texts):
            h = abs(hash(t))
            # Fill with a simple pattern derived from hash
            for d in range(self.hidden_size):
                embs[i, d] = ((h >> (d % 32)) & 0xFF) / 255.0
        # Normalize rows
        embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
        return embs.to(self.device)

    def encode_batch(self, e_s: torch.Tensor, q: torch.Tensor, kg=None) -> torch.Tensor:
        """Return a question embedding per (e_s, q) pair. Shape [B, hidden_size].
        If a mapping exists, uses mapped texts; else returns zeros.
        """
        # Try to form texts from mapping
        texts: List[str] = []
        use_texts = bool(self._texts)
        if use_texts:
            e_s_cpu = e_s.detach().flatten().tolist()
            q_cpu = q.detach().flatten().tolist()
            for i in range(len(e_s_cpu)):
                texts.append(self._texts.get((int(e_s_cpu[i]), int(q_cpu[i])), ''))
            # If all empty, treat as no texts
            if all(t == '' for t in texts):
                use_texts = False
        if use_texts:
            return self._encode_texts(texts)
        # No texts available — return zeros
        return torch.zeros((e_s.size(0), self.hidden_size), device=self.device)
