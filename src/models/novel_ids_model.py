
# STEP 6: NOVEL-IDS-V2 COMPLETE MODEL ARCHITECTURE

import torch
import torch.nn as nn


class SharedFeatureEncoder(nn.Module):
    """Encodes common numerical and categorical network features."""

    def __init__(
        self,
        categorical_vocab_sizes,
        embedding_dim=64,
        attention_heads=4,
        transformer_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.numeric_projection = nn.Linear(1, embedding_dim)

        self.protocol_embedding = nn.Embedding(
            categorical_vocab_sizes["protocol"], embedding_dim
        )
        self.service_embedding = nn.Embedding(
            categorical_vocab_sizes["service"], embedding_dim
        )

        self.feature_identity = nn.Embedding(5, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, shared_numeric, shared_categorical):
        numeric_tokens = self.numeric_projection(
            shared_numeric.unsqueeze(-1)
        )

        protocol_token = self.protocol_embedding(
            shared_categorical[:, 0]
        ).unsqueeze(1)

        service_token = self.service_embedding(
            shared_categorical[:, 1]
        ).unsqueeze(1)

        tokens = torch.cat(
            [numeric_tokens, protocol_token, service_token],
            dim=1
        )

        feature_ids = torch.arange(5, device=tokens.device)
        tokens = tokens + self.feature_identity(feature_ids).unsqueeze(0)

        return self.layer_norm(self.transformer(tokens))


class DatasetSpecificAdapter(nn.Module):
    """Processes features specific to one dataset."""

    def __init__(
        self,
        num_numeric_features,
        num_categories,
        embedding_dim=64,
        dropout=0.1
    ):
        super().__init__()

        self.numeric_projection = nn.Linear(1, embedding_dim)

        self.category_embedding = nn.Embedding(
            num_categories, embedding_dim
        )

        self.feature_identity = nn.Embedding(
            num_numeric_features + 1, embedding_dim
        )

        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, specific_numeric, specific_categorical):
        numeric_tokens = self.numeric_projection(
            specific_numeric.unsqueeze(-1)
        )

        category_token = self.category_embedding(
            specific_categorical[:, 0]
        ).unsqueeze(1)

        tokens = torch.cat([numeric_tokens, category_token], dim=1)

        feature_ids = torch.arange(tokens.size(1), device=tokens.device)
        tokens = tokens + self.feature_identity(feature_ids).unsqueeze(0)

        adapted_tokens = self.adapter(tokens)

        return self.layer_norm(tokens + adapted_tokens)


class AdaptiveResidualFusion(nn.Module):
    """Learns how to combine shared and dataset-specific features."""

    def __init__(self, embedding_dim=64):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, shared_tokens, residual_tokens):
        shared_summary = shared_tokens.mean(dim=1, keepdim=True)

        shared_expanded = shared_summary.expand(
            -1, residual_tokens.size(1), -1
        )

        combined = torch.cat(
            [shared_expanded, residual_tokens], dim=-1
        )

        gates = self.gate(combined)

        fused_tokens = (
            gates * shared_expanded
            + (1 - gates) * residual_tokens
        )

        return self.layer_norm(fused_tokens), gates


class FinalTransformerClassifier(nn.Module):
    """Produces normal-versus-attack predictions."""

    def __init__(
        self,
        embedding_dim=64,
        attention_heads=4,
        transformer_layers=2,
        dropout=0.1,
        num_classes=2
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, shared_tokens, fused_tokens):
        tokens = torch.cat([shared_tokens, fused_tokens], dim=1)

        encoded = self.layer_norm(self.transformer(tokens))

        pooled = encoded.mean(dim=1)

        return self.classifier(pooled)


class NOVELIDSModel(nn.Module):
    """Complete shared encoder + adapters + fusion + classifier."""

    def __init__(
        self,
        categorical_vocab_sizes,
        unsw_specific_numeric_features,
        nsl_specific_numeric_features,
        unsw_category_count,
        nsl_category_count,
        embedding_dim=64,
        attention_heads=4,
        transformer_layers=2,
        dropout=0.1
    ):
        super().__init__()

        self.shared_encoder = SharedFeatureEncoder(
            categorical_vocab_sizes,
            embedding_dim,
            attention_heads,
            transformer_layers,
            dropout
        )

        self.adapters = nn.ModuleDict({
            "UNSW": DatasetSpecificAdapter(
                unsw_specific_numeric_features,
                unsw_category_count,
                embedding_dim,
                dropout
            ),
            "NSL": DatasetSpecificAdapter(
                nsl_specific_numeric_features,
                nsl_category_count,
                embedding_dim,
                dropout
            )
        })

        self.fusions = nn.ModuleDict({
            "UNSW": AdaptiveResidualFusion(embedding_dim),
            "NSL": AdaptiveResidualFusion(embedding_dim)
        })

        self.classifier = FinalTransformerClassifier(
            embedding_dim,
            attention_heads,
            transformer_layers,
            dropout
        )

    def forward(
        self,
        shared_numeric,
        shared_categorical,
        specific_numeric,
        specific_categorical,
        dataset
    ):
        if dataset not in self.adapters:
            raise ValueError(
                "Dataset must be 'UNSW' or 'NSL'."
            )

        shared_tokens = self.shared_encoder(
            shared_numeric,
            shared_categorical
        )

        residual_tokens = self.adapters[dataset](
            specific_numeric,
            specific_categorical
        )

        fused_tokens, gates = self.fusions[dataset](
            shared_tokens,
            residual_tokens
        )

        logits = self.classifier(shared_tokens, fused_tokens)

        return logits, shared_tokens, gates
