"""
ViLBERT Implementation Matching Facebook's Exact Architecture

This implementation exactly matches Facebook's pretrained ViLBERT weights from:
https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin

Architecture:
- bert.embeddings: Text embeddings (from BERT)
- bert.v_embeddings: Visual embeddings (2048 -> 1024)
- bert.encoder.layer: 12 text transformer layers (768-dim)
- bert.encoder.v_layer: 6 visual transformer layers (1024-dim)
- bert.encoder.c_layer: 6 co-attention layers
- bert.t_pooler: Text pooler (768 -> 1024)
- bert.v_pooler: Visual pooler (1024 -> 1024)

Co-attention structure:
- biattention: query1/key1/value1 (visual), query2/key2/value2 (text)
- biOutput: dense1/dense2 (visual->text), q_dense1/q_dense2 (text->visual)
- v_intermediate, v_output: Visual FFN after co-attention
- t_intermediate, t_output: Text FFN after co-attention
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel

logger = logging.getLogger(__name__)


def get_facebook_vilbert_config() -> Dict[str, Any]:
    """Configuration matching Facebook's ViLBERT."""
    return {
        # Text (BERT) config
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "vocab_size": 30522,
        # Visual config
        "v_hidden_size": 1024,
        "v_num_attention_heads": 8,
        "v_num_hidden_layers": 6,
        "v_intermediate_size": 1024,
        "v_hidden_dropout_prob": 0.1,
        "v_attention_probs_dropout_prob": 0.1,
        # Co-attention
        "num_co_attention_layers": 6,
        "bi_hidden_size": 1024,
        # Visual input
        "v_feature_size": 2048,
        "v_loc_size": 5,
    }


class BertLayerNorm(nn.Module):
    """LayerNorm matching BERT/Facebook implementation."""

    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias


class VisualEmbeddings(nn.Module):
    """
    Visual embeddings matching Facebook's bert.v_embeddings structure.

    Keys:
    - image_embeddings.weight/bias: [1024, 2048]
    - image_location_embeddings.weight/bias: [1024, 5]
    - LayerNorm.weight/bias: [1024]
    """

    def __init__(self, config):
        super().__init__()
        self.image_embeddings = nn.Linear(
            config["v_feature_size"], config["v_hidden_size"]
        )
        self.image_location_embeddings = nn.Linear(
            config["v_loc_size"], config["v_hidden_size"]
        )
        self.LayerNorm = BertLayerNorm(config["v_hidden_size"])
        self.dropout = nn.Dropout(config["v_hidden_dropout_prob"])

    def forward(self, visual_features, spatial_locations):
        img_emb = self.image_embeddings(visual_features)
        loc_emb = self.image_location_embeddings(spatial_locations)
        embeddings = self.LayerNorm(img_emb + loc_emb)
        return self.dropout(embeddings)


class BertSelfAttention(nn.Module):
    """Standard BERT self-attention."""

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    """BERT self-attention output."""

    def __init__(self, hidden_size, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """BERT attention = self-attention + output."""

    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super().__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = BertSelfOutput(hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertIntermediate(nn.Module):
    """BERT intermediate (FFN first layer)."""

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class BertOutput(nn.Module):
    """BERT output (FFN second layer + residual)."""

    def __init__(self, intermediate_size, hidden_size, dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    """Standard BERT transformer layer."""

    def __init__(
        self, hidden_size, num_attention_heads, intermediate_size, dropout_prob
    ):
        super().__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.output = BertOutput(intermediate_size, hidden_size, dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BiAttention(nn.Module):
    """
    Bidirectional co-attention matching Facebook's biattention structure.

    Keys:
    - query1/key1/value1: Visual stream (1024 -> 1024)
    - query2/key2/value2: Text stream (768 -> 1024)
    """

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config["v_num_attention_heads"]
        self.attention_head_size = config["bi_hidden_size"] // self.num_attention_heads

        # Visual stream: 1024 -> 1024
        self.query1 = nn.Linear(config["v_hidden_size"], config["bi_hidden_size"])
        self.key1 = nn.Linear(config["v_hidden_size"], config["bi_hidden_size"])
        self.value1 = nn.Linear(config["v_hidden_size"], config["bi_hidden_size"])

        # Text stream: 768 -> 1024
        self.query2 = nn.Linear(config["hidden_size"], config["bi_hidden_size"])
        self.key2 = nn.Linear(config["hidden_size"], config["bi_hidden_size"])
        self.value2 = nn.Linear(config["hidden_size"], config["bi_hidden_size"])

        self.dropout = nn.Dropout(config["attention_probs_dropout_prob"])

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, v_hidden, t_hidden, v_attention_mask=None, t_attention_mask=None):
        """
        Cross-attention between visual and text streams.

        Returns:
            v_context: Visual attended by text
            t_context: Text attended by visual
        """
        # Visual queries attend to text keys/values
        v_query = self.transpose_for_scores(self.query1(v_hidden))
        t_key = self.transpose_for_scores(self.key2(t_hidden))
        t_value = self.transpose_for_scores(self.value2(t_hidden))

        # Text queries attend to visual keys/values
        t_query = self.transpose_for_scores(self.query2(t_hidden))
        v_key = self.transpose_for_scores(self.key1(v_hidden))
        v_value = self.transpose_for_scores(self.value1(v_hidden))

        # Visual attending to text
        v_scores = torch.matmul(v_query, t_key.transpose(-1, -2))
        v_scores = v_scores / math.sqrt(self.attention_head_size)
        if t_attention_mask is not None:
            v_scores = v_scores + t_attention_mask
        v_probs = self.dropout(F.softmax(v_scores, dim=-1))
        v_context = torch.matmul(v_probs, t_value)

        # Text attending to visual
        t_scores = torch.matmul(t_query, v_key.transpose(-1, -2))
        t_scores = t_scores / math.sqrt(self.attention_head_size)
        if v_attention_mask is not None:
            t_scores = t_scores + v_attention_mask
        t_probs = self.dropout(F.softmax(t_scores, dim=-1))
        t_context = torch.matmul(t_probs, v_value)

        # Reshape back
        v_context = v_context.permute(0, 2, 1, 3).contiguous()
        v_context = v_context.view(v_context.size()[:-2] + (-1,))

        t_context = t_context.permute(0, 2, 1, 3).contiguous()
        t_context = t_context.view(t_context.size()[:-2] + (-1,))

        return v_context, t_context


class BiOutput(nn.Module):
    """
    Bidirectional output layer matching Facebook's biOutput structure.

    Keys:
    - dense1/LayerNorm1: Visual output (1024 -> 1024)
    - dense2/LayerNorm2: Text output (1024 -> 768)
    - q_dense1: Visual query transform (1024 -> 1024)
    - q_dense2: Text query transform (1024 -> 768)
    """

    def __init__(self, config):
        super().__init__()
        # Visual output: bi_hidden -> v_hidden
        self.dense1 = nn.Linear(config["bi_hidden_size"], config["v_hidden_size"])
        self.LayerNorm1 = BertLayerNorm(config["v_hidden_size"])

        # Text output: bi_hidden -> hidden
        self.dense2 = nn.Linear(config["bi_hidden_size"], config["hidden_size"])
        self.LayerNorm2 = BertLayerNorm(config["hidden_size"])

        # Query transforms (for residual)
        self.q_dense1 = nn.Linear(config["bi_hidden_size"], config["v_hidden_size"])
        self.q_dense2 = nn.Linear(config["bi_hidden_size"], config["hidden_size"])

        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, v_context, t_context, v_hidden, t_hidden):
        """
        Apply output transforms and residual connections.
        """
        # Visual output with residual
        v_out = self.dense1(v_context)
        v_out = self.dropout(v_out)
        v_out = self.LayerNorm1(v_out + v_hidden)

        # Text output with residual
        t_out = self.dense2(t_context)
        t_out = self.dropout(t_out)
        t_out = self.LayerNorm2(t_out + t_hidden)

        return v_out, t_out


class CoAttentionLayer(nn.Module):
    """
    Co-attention layer matching Facebook's c_layer structure.

    Components:
    - biattention: Cross-modal attention
    - biOutput: Output projection with residual
    - v_intermediate, v_output: Visual FFN
    - t_intermediate, t_output: Text FFN
    """

    def __init__(self, config):
        super().__init__()
        self.biattention = BiAttention(config)
        self.biOutput = BiOutput(config)

        # Visual FFN
        self.v_intermediate = BertIntermediate(
            config["v_hidden_size"], config["v_intermediate_size"]
        )
        self.v_output = BertOutput(
            config["v_intermediate_size"],
            config["v_hidden_size"],
            config["v_hidden_dropout_prob"],
        )

        # Text FFN
        self.t_intermediate = BertIntermediate(
            config["hidden_size"], config["intermediate_size"]
        )
        self.t_output = BertOutput(
            config["intermediate_size"],
            config["hidden_size"],
            config["hidden_dropout_prob"],
        )

    def forward(self, v_hidden, t_hidden, v_attention_mask=None, t_attention_mask=None):
        # Cross-attention
        v_context, t_context = self.biattention(
            v_hidden, t_hidden, v_attention_mask, t_attention_mask
        )

        # Output projection
        v_attn_out, t_attn_out = self.biOutput(v_context, t_context, v_hidden, t_hidden)

        # Visual FFN
        v_inter = self.v_intermediate(v_attn_out)
        v_out = self.v_output(v_inter, v_attn_out)

        # Text FFN
        t_inter = self.t_intermediate(t_attn_out)
        t_out = self.t_output(t_inter, t_attn_out)

        return v_out, t_out


class BertPooler(nn.Module):
    """Pooler for extracting [CLS] representation."""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)

    def forward(self, hidden_states):
        # Take first token ([CLS] for text, first region for visual)
        first_token = hidden_states[:, 0]
        pooled = torch.tanh(self.dense(first_token))
        return pooled


class ViLBERTEncoder(nn.Module):
    """
    ViLBERT encoder matching Facebook's structure.

    Architecture:
    - layer: 12 text transformer layers
    - v_layer: 6 visual transformer layers
    - c_layer: 6 co-attention layers

    The co-attention is applied after specific text layers.
    """

    def __init__(self, config):
        super().__init__()
        # Text layers (12)
        self.layer = nn.ModuleList(
            [
                BertLayer(
                    config["hidden_size"],
                    config["num_attention_heads"],
                    config["intermediate_size"],
                    config["hidden_dropout_prob"],
                )
                for _ in range(config["num_hidden_layers"])
            ]
        )

        # Visual layers (6)
        self.v_layer = nn.ModuleList(
            [
                BertLayer(
                    config["v_hidden_size"],
                    config["v_num_attention_heads"],
                    config["v_intermediate_size"],
                    config["v_hidden_dropout_prob"],
                )
                for _ in range(config["v_num_hidden_layers"])
            ]
        )

        # Co-attention layers (6)
        self.c_layer = nn.ModuleList(
            [CoAttentionLayer(config) for _ in range(config["num_co_attention_layers"])]
        )

        # Co-attention is applied after text layers: 1, 3, 5, 7, 9, 11
        self.co_attention_text_layers = [1, 3, 5, 7, 9, 11]

    def forward(self, t_hidden, v_hidden, t_attention_mask=None, v_attention_mask=None):
        v_layer_idx = 0
        c_layer_idx = 0

        for t_layer_idx, text_layer in enumerate(self.layer):
            # Text self-attention
            t_hidden = text_layer(t_hidden, t_attention_mask)

            # Apply co-attention after specific text layers
            if t_layer_idx in self.co_attention_text_layers and c_layer_idx < len(
                self.c_layer
            ):
                # Visual self-attention first
                v_hidden = self.v_layer[v_layer_idx](v_hidden, v_attention_mask)
                v_layer_idx += 1

                # Then co-attention
                v_hidden, t_hidden = self.c_layer[c_layer_idx](
                    v_hidden, t_hidden, v_attention_mask, t_attention_mask
                )
                c_layer_idx += 1

        return t_hidden, v_hidden


class ViLBERTModel(nn.Module):
    """
    ViLBERT model matching Facebook's bert.* structure.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Text embeddings (from BERT)
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
        )
        bert = BertModel(bert_config)
        self.embeddings = bert.embeddings

        # Visual embeddings
        self.v_embeddings = VisualEmbeddings(config)

        # Encoder
        self.encoder = ViLBERTEncoder(config)

        # Poolers
        self.t_pooler = BertPooler(config["hidden_size"], config["bi_hidden_size"])
        self.v_pooler = BertPooler(config["v_hidden_size"], config["v_hidden_size"])

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_features=None,
        visual_attention_mask=None,
        spatial_locations=None,
    ):
        # Text embeddings
        t_hidden = self.embeddings(input_ids, token_type_ids=token_type_ids)

        # Visual embeddings
        v_hidden = self.v_embeddings(visual_features, spatial_locations)

        # Prepare attention masks
        if attention_mask is not None:
            extended_t_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_t_mask = (1.0 - extended_t_mask) * -10000.0
        else:
            extended_t_mask = None

        if visual_attention_mask is not None:
            extended_v_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_v_mask = (1.0 - extended_v_mask) * -10000.0
        else:
            extended_v_mask = None

        # Encode
        t_hidden, v_hidden = self.encoder(
            t_hidden, v_hidden, extended_t_mask, extended_v_mask
        )

        # Pool
        t_pooled = self.t_pooler(t_hidden)
        v_pooled = self.v_pooler(v_hidden)

        return t_hidden, v_hidden, t_pooled, v_pooled


class ViLBERTForClassification(nn.Module):
    """
    ViLBERT for binary classification (Hateful Memes).
    """

    def __init__(self, config, num_labels=2):
        super().__init__()
        self.config = config
        self.num_labels = num_labels

        # ViLBERT backbone - named 'bert' to match Facebook's checkpoint
        self.bert = ViLBERTModel(config)

        # Classification head
        # Concatenate text and visual pooled outputs
        classifier_input_size = (
            config["bi_hidden_size"] + config["v_hidden_size"]
        )  # 1024 + 1024 = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input_size, config["bi_hidden_size"]),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config["bi_hidden_size"], num_labels),
        )

    def get_num_parameters(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        visual_features=None,
        visual_attention_mask=None,
        spatial_locations=None,
        labels=None,
    ):
        t_hidden, v_hidden, t_pooled, v_pooled = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            spatial_locations=spatial_locations,
        )

        # Concatenate pooled representations
        pooled = torch.cat([t_pooled, v_pooled], dim=-1)

        # Classify
        logits = self.classifier(pooled)

        outputs = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            outputs["loss"] = loss_fn(logits, labels)

        return outputs


def load_facebook_weights(model: ViLBERTForClassification, checkpoint_path: str) -> int:
    """
    Load Facebook's pretrained weights into the model.

    Returns number of loaded weight tensors.
    """
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    model_state = model.state_dict()
    new_state_dict = {}
    loaded = 0

    for fb_key, fb_value in state_dict.items():
        # Facebook keys start with 'bert.' or 'cls.'
        # Our model has 'bert.' at the top level

        if fb_key in model_state:
            if model_state[fb_key].shape == fb_value.shape:
                new_state_dict[fb_key] = fb_value
                loaded += 1
        else:
            # Try without 'bert.' prefix for embeddings in our structure
            # Facebook: bert.embeddings -> our: bert.embeddings (should match)
            pass

    model.load_state_dict(new_state_dict, strict=False)

    logger.info(f"Loaded {loaded}/{len(state_dict)} Facebook weights")

    # Report unmatched
    fb_keys = set(state_dict.keys())
    matched_keys = set(new_state_dict.keys())
    unmatched = fb_keys - matched_keys
    if unmatched:
        # Filter out cls.* keys (pretraining heads we don't need)
        unmatched_important = [k for k in unmatched if not k.startswith("cls.")]
        if unmatched_important:
            logger.warning(f"Unmatched important keys: {unmatched_important[:10]}")

    return loaded
