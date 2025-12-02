"""
ViLBERT: Vision-and-Language BERT
Two-stream architecture with co-attentional transformer layers.

Based on: "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations
for Vision-and-Language Tasks" (Lu et al., NeurIPS 2019)

Architecture Overview:
- Two parallel transformer streams: one for vision, one for language
- Co-attention layers enable cross-modal interaction at specific layers
- Visual input: Faster R-CNN region features (2048-dim, 36 regions typical)
- Text input: BERT tokenized text
"""

import copy
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer


class BertLayerNorm(nn.Module):
    """LayerNorm module in the TF style (epsilon inside the square root)."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertSelfAttention(nn.Module):
    """Standard BERT self-attention mechanism."""

    def __init__(self, config: Dict):
        super().__init__()
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.get("attention_probs_dropout_prob", 0.1))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
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

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class BertCoAttention(nn.Module):
    """
    Co-Attention mechanism for cross-modal interaction.
    Query comes from one modality, Key/Value from the other.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.num_attention_heads = config.get("num_attention_heads", 12)
        self.hidden_size = config.get("hidden_size", 768)
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.get("attention_probs_dropout_prob", 0.1))

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states_query: torch.Tensor,  # From one modality
        hidden_states_key: torch.Tensor,  # From the other modality
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Cross-modal attention: Query attends to Key/Value from other modality.
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states_query))
        key_layer = self.transpose_for_scores(self.key(hidden_states_key))
        value_layer = self.transpose_for_scores(self.value(hidden_states_key))

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

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )
        return outputs


class BertSelfOutput(nn.Module):
    """Output projection + residual + LayerNorm for self-attention."""

    def __init__(self, config: Dict):
        super().__init__()
        hidden_size = config.get("hidden_size", 768)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertIntermediate(nn.Module):
    """Feed-forward intermediate layer (expand dimension)."""

    def __init__(self, config: Dict):
        super().__init__()
        hidden_size = config.get("hidden_size", 768)
        intermediate_size = config.get("intermediate_size", 3072)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """Feed-forward output layer (compress dimension back)."""

    def __init__(self, config: Dict):
        super().__init__()
        hidden_size = config.get("hidden_size", 768)
        intermediate_size = config.get("intermediate_size", 3072)
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """Full attention block: self-attention + output projection."""

    def __init__(self, config: Dict):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertCrossAttention(nn.Module):
    """Full cross-attention block for co-attention."""

    def __init__(self, config: Dict):
        super().__init__()
        self.self = BertCoAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states_query: torch.Tensor,
        hidden_states_key: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        self_outputs = self.self(
            hidden_states_query, hidden_states_key, attention_mask, output_attentions
        )
        attention_output = self.output(self_outputs[0], hidden_states_query)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayer(nn.Module):
    """Single BERT transformer layer: attention + FFN."""

    def __init__(self, config: Dict):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        attention_outputs = self.attention(
            hidden_states, attention_mask, output_attentions
        )
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]
        return outputs


class BertConnectionLayer(nn.Module):
    """
    Co-attention connection layer for cross-modal interaction.
    Performs bidirectional attention: visual->text and text->visual.
    """

    def __init__(self, config: Dict):
        super().__init__()
        # Cross-attention: visual queries attend to text
        self.biattention_v = BertCrossAttention(config)
        # Cross-attention: text queries attend to visual
        self.biattention_t = BertCrossAttention(config)

        # FFN for visual stream
        self.intermediate_v = BertIntermediate(config)
        self.output_v = BertOutput(config)

        # FFN for text stream
        self.intermediate_t = BertIntermediate(config)
        self.output_t = BertOutput(config)

    def forward(
        self,
        visual_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional cross-modal attention.

        Args:
            visual_hidden: Visual features [batch, num_regions, hidden_size]
            text_hidden: Text features [batch, seq_len, hidden_size]
            visual_attention_mask: Mask for visual regions
            text_attention_mask: Mask for text tokens

        Returns:
            Updated (visual_hidden, text_hidden) after cross-modal interaction
        """
        # Visual attends to text (visual is query, text is key/value)
        visual_attention_output = self.biattention_v(
            visual_hidden, text_hidden, text_attention_mask, output_attentions
        )[0]

        # Text attends to visual (text is query, visual is key/value)
        text_attention_output = self.biattention_t(
            text_hidden, visual_hidden, visual_attention_mask, output_attentions
        )[0]

        # FFN for visual
        visual_intermediate = self.intermediate_v(visual_attention_output)
        visual_output = self.output_v(visual_intermediate, visual_attention_output)

        # FFN for text
        text_intermediate = self.intermediate_t(text_attention_output)
        text_output = self.output_t(text_intermediate, text_attention_output)

        return visual_output, text_output


class ViLBERTEncoder(nn.Module):
    """
    ViLBERT Encoder with two parallel transformer streams connected via co-attention.

    Architecture:
    - Visual stream: 6 transformer layers (default)
    - Text stream: 12 transformer layers (BERT-base)
    - Co-attention connections at specified layer indices
    """

    def __init__(self, config: Dict):
        super().__init__()

        # Configuration
        self.v_num_layers = config.get("v_num_hidden_layers", 6)  # Visual layers
        self.t_num_layers = config.get("t_num_hidden_layers", 12)  # Text layers
        self.num_co_layers = config.get(
            "num_co_layers", 6
        )  # Number of co-attention connections

        # Visual transformer layers
        self.v_layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.v_num_layers)]
        )

        # Text transformer layers
        self.t_layer = nn.ModuleList(
            [BertLayer(config) for _ in range(self.t_num_layers)]
        )

        # Co-attention connection layers
        self.c_layer = nn.ModuleList(
            [BertConnectionLayer(config) for _ in range(self.num_co_layers)]
        )

        # Determine which layers have co-attention
        # Default: connect at layers [1, 3, 5, 7, 9, 11] for text (every other layer)
        self.v_start_layer = config.get("v_start_layer", 0)

    def forward(
        self,
        visual_hidden: torch.Tensor,
        text_hidden: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ViLBERT encoder.

        Visual and text streams process in parallel, with periodic co-attention.
        """
        v_idx = 0
        co_idx = 0

        for t_idx in range(self.t_num_layers):
            # Process text layer
            text_outputs = self.t_layer[t_idx](
                text_hidden, text_attention_mask, output_attentions
            )
            text_hidden = text_outputs[0]

            # Check if we should do co-attention at this layer
            # Co-attention happens every 2 text layers, starting from layer 1
            if (t_idx + 1) % 2 == 0 and co_idx < self.num_co_layers:
                # Process visual layer first
                if v_idx < self.v_num_layers:
                    visual_outputs = self.v_layer[v_idx](
                        visual_hidden, visual_attention_mask, output_attentions
                    )
                    visual_hidden = visual_outputs[0]
                    v_idx += 1

                # Then do co-attention
                visual_hidden, text_hidden = self.c_layer[co_idx](
                    visual_hidden,
                    text_hidden,
                    visual_attention_mask,
                    text_attention_mask,
                    output_attentions,
                )
                co_idx += 1

        return visual_hidden, text_hidden


class ViLBERTEmbeddings(nn.Module):
    """
    Embeddings for visual features.
    Projects Faster R-CNN features to BERT hidden size and adds position embeddings.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.v_feature_size = config.get(
            "v_feature_size", 2048
        )  # Faster R-CNN output dim
        self.max_regions = config.get(
            "max_regions", 100
        )  # Max number of visual regions

        # Project visual features to hidden size
        self.image_embeddings = nn.Linear(self.v_feature_size, self.hidden_size)

        # Spatial location embeddings (5-dim: normalized bbox coords)
        self.location_embeddings = nn.Linear(5, self.hidden_size)

        # Position embeddings for visual regions
        self.position_embeddings = nn.Embedding(self.max_regions, self.hidden_size)

        self.LayerNorm = BertLayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))

    def forward(
        self,
        visual_features: torch.Tensor,
        spatial_locations: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, num_regions, v_feature_size] - Faster R-CNN features
            spatial_locations: [batch, num_regions, 5] - Normalized bbox coordinates
            position_ids: [batch, num_regions] - Position indices
        """
        batch_size, num_regions, _ = visual_features.shape

        # Project visual features
        visual_embeddings = self.image_embeddings(visual_features)

        # Add spatial location embeddings if provided
        if spatial_locations is not None:
            location_embeddings = self.location_embeddings(spatial_locations)
            visual_embeddings = visual_embeddings + location_embeddings

        # Add position embeddings
        if position_ids is None:
            position_ids = torch.arange(num_regions, device=visual_features.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        position_embeddings = self.position_embeddings(position_ids)
        visual_embeddings = visual_embeddings + position_embeddings

        visual_embeddings = self.LayerNorm(visual_embeddings)
        visual_embeddings = self.dropout(visual_embeddings)

        return visual_embeddings


class ViLBERTModel(nn.Module):
    """
    Complete ViLBERT Model.

    Two-stream vision-language transformer with:
    - BERT for text encoding
    - Visual transformer for image region encoding
    - Co-attention layers for cross-modal interaction
    """

    def __init__(self, config: Dict, bert_model_name: str = "bert-base-uncased"):
        super().__init__()
        self.config = config

        # Text stream: Use pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert_config = self.bert.config

        # Update config with BERT settings
        config["hidden_size"] = self.bert_config.hidden_size
        config["num_attention_heads"] = self.bert_config.num_attention_heads
        config["intermediate_size"] = self.bert_config.intermediate_size
        config["hidden_dropout_prob"] = self.bert_config.hidden_dropout_prob
        config["attention_probs_dropout_prob"] = (
            self.bert_config.attention_probs_dropout_prob
        )

        # Visual embeddings
        self.visual_embeddings = ViLBERTEmbeddings(config)

        # ViLBERT encoder (co-attention layers)
        self.encoder = ViLBERTEncoder(config)

        # Pooler for classification
        self.t_pooler = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]), nn.Tanh()
        )
        self.v_pooler = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]), nn.Tanh()
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        visual_features: torch.Tensor = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        spatial_locations: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ViLBERT.

        Args:
            input_ids: Text token IDs [batch, seq_len]
            attention_mask: Text attention mask [batch, seq_len]
            token_type_ids: Text segment IDs [batch, seq_len]
            visual_features: Faster R-CNN region features [batch, num_regions, 2048]
            visual_attention_mask: Visual region mask [batch, num_regions]
            spatial_locations: Normalized bbox coords [batch, num_regions, 5]

        Returns:
            Dictionary with pooled outputs and sequence outputs
        """
        # Get BERT embeddings (not full forward - we'll use custom encoder)
        bert_outputs = self.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        text_hidden = bert_outputs

        # Get visual embeddings
        visual_hidden = self.visual_embeddings(visual_features, spatial_locations)

        # Prepare attention masks (convert to additive mask)
        if attention_mask is not None:
            # [batch, seq_len] -> [batch, 1, 1, seq_len]
            extended_text_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_text_mask = (1.0 - extended_text_mask) * -10000.0
        else:
            extended_text_mask = None

        if visual_attention_mask is not None:
            extended_visual_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_visual_mask = (1.0 - extended_visual_mask) * -10000.0
        else:
            extended_visual_mask = None

        # Pass through ViLBERT encoder
        visual_output, text_output = self.encoder(
            visual_hidden,
            text_hidden,
            extended_visual_mask,
            extended_text_mask,
            output_attentions,
        )

        # Pool outputs (use [CLS] token for text, mean for visual)
        text_pooled = self.t_pooler(text_output[:, 0])  # [CLS] token
        visual_pooled = self.v_pooler(visual_output.mean(dim=1))  # Mean pool

        return {
            "text_output": text_output,
            "visual_output": visual_output,
            "text_pooled": text_pooled,
            "visual_pooled": visual_pooled,
            "pooled_output": torch.cat([text_pooled, visual_pooled], dim=-1),
        }


class ViLBERTForClassification(nn.Module):
    """
    ViLBERT with classification head for Hateful Memes detection.
    """

    def __init__(
        self,
        config: Dict,
        num_labels: int = 2,
        bert_model_name: str = "bert-base-uncased",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.vilbert = ViLBERTModel(config, bert_model_name)

        hidden_size = config.get("hidden_size", 768)
        classifier_dropout = config.get("classifier_dropout", 0.5)

        # Classification head (takes concatenated pooled outputs)
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        visual_features: torch.Tensor = None,
        visual_attention_mask: Optional[torch.Tensor] = None,
        spatial_locations: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            labels: Ground truth labels [batch] for computing loss

        Returns:
            Dictionary with logits and optionally loss
        """
        outputs = self.vilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_features=visual_features,
            visual_attention_mask=visual_attention_mask,
            spatial_locations=spatial_locations,
        )

        pooled_output = outputs["pooled_output"]
        logits = self.classifier(pooled_output)

        result = {"logits": logits, "pooled_output": pooled_output, **outputs}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        return F.softmax(logits, dim=-1)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        return torch.argmax(logits, dim=-1)


def get_vilbert_config() -> Dict:
    """Get default ViLBERT configuration for Hateful Memes."""
    return {
        # Model architecture
        "hidden_size": 768,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        # Visual stream
        "v_feature_size": 2048,  # Faster R-CNN output dimension
        "v_num_hidden_layers": 6,  # Visual transformer layers
        "max_regions": 100,  # Max visual regions
        # Text stream
        "t_num_hidden_layers": 12,  # BERT layers
        # Co-attention
        "num_co_layers": 6,  # Number of co-attention connections
        # Classifier
        "classifier_dropout": 0.5,
        "num_labels": 2,
    }


if __name__ == "__main__":
    # Test the model
    config = get_vilbert_config()
    model = ViLBERTForClassification(config, num_labels=2)

    # Create dummy inputs
    batch_size = 2
    seq_len = 128
    num_regions = 36

    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    visual_features = torch.randn(batch_size, num_regions, 2048)
    visual_attention_mask = torch.ones(batch_size, num_regions)
    labels = torch.tensor([0, 1])

    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_features=visual_features,
        visual_attention_mask=visual_attention_mask,
        labels=labels,
    )

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Pooled output shape: {outputs['pooled_output'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
