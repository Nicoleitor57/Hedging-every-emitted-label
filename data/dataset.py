"""
data/dataset.py

Dataset definition for end-to-end span detection and classification.
Handles JSON parsing, tokenization, and offset alignment.
"""

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

class JointDataset(Dataset):
    """
    PyTorch Dataset for Joint Span Detection and Classification.
    Parses clinical notes and their annotated entities, aligning character 
    offsets to token-level coordinates.
    """
    def __init__(self, json_source, dict_path, tokenizer, max_length=512, l_max=110,
                 exclude_vids=None, long_max_length=None):
        """
        Args:
            json_source (str, list, Path): Path to a JSON file, directory of JSONs, or list of paths.
            dict_path (str): Path to the dictionary mapping variable IDs to hierarchical paths.
            tokenizer (PreTrainedTokenizer): HuggingFace fast tokenizer (requires offset_mapping).
            max_length (int): Maximum sequence length for the tokenizer.
            l_max (int): Maximum span length to consider for the detector target matrix.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.l_max = l_max
        # VENTANA DESLIZANTE (2026-10-17; main.py --long-max-length). None = apagado, todo idéntico. Con un
        # tope, una nota de más de `max_length` tokens ya no se trunca: se tokeniza entera (hasta el tope) y
        # se rellena a un múltiplo de 64, y el encoder la recorre por ventanas (models/encoders.py
        # encode_windowed). Una nota que cabe en `max_length` sale exactamente igual que siempre.
        self.long_max_length = int(long_max_length) if long_max_length else None

        # Load label dictionary and create a sorted mapping of ID -> Index
        with open(dict_path, 'r', encoding='utf-8') as f:
            self.label_dict = json.load(f)
            
        sorted_ids = sorted(self.label_dict.keys(), key=lambda x: int(x))
        self.id_to_idx = {s: i for i, s in enumerate(sorted_ids)}
        valid_var_ids = set(self.label_dict.keys())
        # CLASSES EXCLUDED FROM THE DATA ENTIRELY. Their labels are dropped here,
        # so a span carrying only excluded classes is never created: the detector
        # sees that region as background and the linker never gets a positive for
        # them. Used for the free-text catch-alls (Hallazgos Adicionales y
        # Comentarios, Comportamiento, Extension diagnostico) -- open-ended prose
        # whose boundaries are arbitrary, ~30% of gold at ~54% loss, i.e. noise in
        # the gradient rather than signal.
        # The class INDICES are kept so ids
        # stay stable; the classes simply never occur.
        self.exclude_vids = {str(v) for v in (exclude_vids or ())}
        self._n_excluded_labels = 0

        # Resolve input sources (single file, directory, or list)
        if isinstance(json_source, (list, tuple)):
            paths = [Path(p) for p in json_source]
        else:
            src = Path(json_source)
            if src.is_dir():
                paths = list(src.glob("*.json"))
            elif src.suffix == '.json':
                with open(src) as f:
                    content = json.load(f)
                paths = [Path(p) for p in content] if isinstance(content, list) else [src]
            else:
                raise ValueError(f"Invalid json_source: {json_source}")
                
        paths = [p for p in paths if p.exists()]

        # Parse and store samples in memory
        self.samples = []
        for path in paths:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            text = data.get('note_text', '')
            spans = []
            
            for l in data.get('labels', []):
                s, e = l['start'], l['end']
                var_id = str(l.get('variable_id'))
                
                # Filter out invalid spans or unrecognized entities
                if e <= s:
                    continue
                if var_id not in valid_var_ids:
                    continue
                if var_id in self.exclude_vids:
                    self._n_excluded_labels += 1
                    continue

                spans.append((s, e, var_id))

            self.samples.append({
                'text': text, 
                'spans': spans, 
                'source': path.name
            })

    def __len__(self):
        return len(self.samples)

    def _align(self, offsets, attention_mask, cs, ce):
        """
        Aligns character-level coordinates (cs, ce) to token-level indices.
        
        Args:
            offsets (torch.Tensor): Offset mappings returned by the fast tokenizer.
            attention_mask (torch.Tensor): Tokenizer attention mask.
            cs (int): Character start index.
            ce (int): Character end index.
            
        Returns:
            tuple: (token_start_idx, token_end_idx) or (None, None) if no overlap is found.
        """
        starts = offsets[:, 0]
        ends = offsets[:, 1]
        
        # Identify tokens that represent actual text (ignores padding and special tokens)
        is_real = (ends > starts) & attention_mask.bool()
        
        # Find tokens that overlap with the character interval
        overlaps = (starts < ce) & (ends > cs) & is_real
        if not overlaps.any():
            return None, None
            
        idx = overlaps.nonzero(as_tuple=False).squeeze(-1)
        return idx[0].item(), idx[-1].item()

    def __getitem__(self, idx):
        """
        Processes a single sample, converting text to tensor inputs and 
        building the detection target matrix.
        """
        sample = self.samples[idx]

        # Largo de la secuencia: `max_length` de siempre, salvo una nota larga con la ventana deslizante.
        N = self.max_length
        if self.long_max_length and self.long_max_length > self.max_length:
            n_tok = len(self.tokenizer(sample['text'], truncation=True,
                                       max_length=self.long_max_length)['input_ids'])
            if n_tok > self.max_length:
                N = min(self.long_max_length, -(-n_tok // 64) * 64)

        # Tokenize with offset mappings (crucial for character-to-token alignment)
        enc = self.tokenizer(
            sample['text'],
            max_length=N,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt',
        )
        
        input_ids = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        offsets = enc['offset_mapping'].squeeze(0)

        # Initialize the sparse target matrix for the biaffine detector [N, l_max]
        det_target = torch.zeros(N, self.l_max)
        gold = []
        grouped = {}
        
        # Process and align each gold span
        for cs, ce, var_id in sample['spans']:
            ts, te = self._align(offsets, attention_mask, cs, ce)
            if ts is None:
                continue
                
            length = te - ts
            
            # Discard spans that exceed the maximum defined length
            if length < 0 or length >= self.l_max:
                continue
                
            # Mark the span in the detection target matrix
            det_target[ts, length] = 1.0

            # Store the token-level boundaries and class index
            gold.append((ts, te, self.id_to_idx[var_id]))
            grouped.setdefault((ts, te), []).append(self.id_to_idx[var_id])

        # Same gold, grouped by OFFSET instead of one row per label.
        #
        # 22.5% of span offsets in this corpus carry more than one class (41% of
        # all gold labels sit on such an offset), so the per-label view hands the
        # linker two rows with IDENTICAL features and DIFFERENT targets -- targets
        # it cannot both satisfy. It also caps single-label recall at
        # 177389/233232 = 76.1% by construction.
        #
        # Emitted ALONGSIDE `gold_spans`, never instead of it, so the single-label
        # baseline stays bit-identical and the two heads are comparable.
        gold_multi = [(ts, te, sorted(set(v))) for (ts, te), v in grouped.items()]

        out = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'det_target': det_target,
            'gold_spans': gold,
            'gold_multi': gold_multi,
        }
        return out


def _apilar(ts, valor=0):
    """torch.stack; si los largos difieren (sólo con la ventana deslizante, donde una nota larga trae su
    propio N) rellena la 1.ª dimensión hasta el máximo del lote. El relleno queda con attention_mask 0,
    así que el valor de los input_ids ahí no lo lee nadie."""
    n = max(t.size(0) for t in ts)
    if all(t.size(0) == n for t in ts):
        return torch.stack(ts)
    return torch.stack([t if t.size(0) == n else
                        torch.cat([t, t.new_full((n - t.size(0),) + tuple(t.shape[1:]), valor)])
                        for t in ts])


def collate_fn(batch):
    """
    Custom collate function to batch variable-length gold span lists.
    """
    return {
        'input_ids':      _apilar([b['input_ids'] for b in batch]),
        'attention_mask': _apilar([b['attention_mask'] for b in batch]),
        'det_target':     _apilar([b['det_target'] for b in batch]),
        'gold_spans':     [b['gold_spans'] for b in batch],  # Kept as list of lists
        'gold_multi':     [b['gold_multi'] for b in batch],  # offset-grouped multi-label
    }