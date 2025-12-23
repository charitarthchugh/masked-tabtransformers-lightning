import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from tab_transformer_pytorch import TabTransformer
from torch import nn


class TabTransformerModuleforMLM(L.LightningModule):
    def __init__(
        self,
        categorical_unique_counts: list[int],
        dim: int = 32,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        num_continuous=0,
        mlp_act: nn.Module = None,
        continuous_mean_std=None,
        return_attention: bool = False,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        if not mlp_act:
            mlp_act = nn.ReLU()
        self.return_attention = return_attention
        self.learning_rate = learning_rate
        # model instantiation copied from example
        self.model = TabTransformer(
            categories=categorical_unique_counts,
            num_continuous=num_continuous,  # number of continuous values
            dim=dim,  # dimension, paper set at 32
            dim_out=1,  # We'll add our own heads, so set to 1 to minimize MLP overhead
            depth=depth,  # depth, paper recommended 6
            heads=heads,  # heads, paper recommends 8
            attn_dropout=attn_dropout,  # post-attention dropout
            ff_dropout=ff_dropout,  # feed forward dropout
            mlp_hidden_mults=mlp_hidden_mults,  # relative multiples of each hidden dimension of the last mlp to logits
            mlp_act=mlp_act,  # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            use_shared_categ_embed=False,
            num_special_tokens=2,
            continuous_mean_std=continuous_mean_std,  # (optional) - normalize the continuous values before layer norm
        )

        # Create separate classifier heads for each categorical feature
        # Each head predicts the original value for that feature
        self.categorical_unique_counts = categorical_unique_counts
        self.num_categorical = len(categorical_unique_counts)

        # Get the actual embedding dimension from transformer output
        # This will be: num_categorical * dim + num_continuous (after flattening transformer output)
        mlp_input_dim = len(categorical_unique_counts) * dim + num_continuous

        # Create a classifier head for each categorical feature
        self.classifier_heads = nn.ModuleList([
            nn.Linear(mlp_input_dim, count) for count in categorical_unique_counts
        ])

        print(self.model)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, masked_categorical, masked_continuous):
        """
        Forward pass for MLM.
        Returns logits for each categorical feature.

        Uses -1 for masked positions, which TabTransformer handles as special tokens.
        """
        # Replicate TabTransformer forward but extract intermediate representation
        # Based on tab_transformer_pytorch.py source code

        xs = []

        # Get categorical embeddings
        is_special_token = masked_categorical < 0

        # Clamp to valid range: [0, num_classes-1] for each feature
        # masked_categorical shape: [batch, num_features]
        masked_cat_clamped = masked_categorical.clone()
        for feat_idx, num_classes in enumerate(self.categorical_unique_counts):
            # Clamp each feature to [0, num_classes-1]
            masked_cat_clamped[:, feat_idx] = masked_categorical[:, feat_idx].clamp(0, num_classes - 1)

        categ_embed = self.model.categorical_embeds(masked_cat_clamped, sum_discrete_sets=False)

        # Replace special token positions with special token embeddings
        if is_special_token.any():
            special_token_ids = (masked_categorical + 1).abs().clamp_max(self.model.num_special_tokens - 1)
            special_embed = self.model.special_token_embed(special_token_ids)
            categ_embed = torch.where(is_special_token[..., None], special_embed, categ_embed)

        # Apply transformer
        x_trans, _ = self.model.transformer(categ_embed, return_attn=True)

        # Flatten categorical embeddings
        flat_categ = x_trans.flatten(1)
        xs.append(flat_categ)

        # Add continuous features
        if self.model.num_continuous > 0:
            x_cont = masked_continuous
            if hasattr(self.model, 'continuous_mean_std') and self.model.continuous_mean_std is not None:
                mean, std = self.model.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.model.norm(x_cont)
            xs.append(normed_cont)

        # Concatenate all features
        x = torch.cat(xs, dim=-1)

        # Apply our classifier heads instead of the model's MLP
        logits = [head(x) for head in self.classifier_heads]

        return logits

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original_categorical = batch["original_categorical"]
        mask = batch["mask"]

        # Forward pass - returns list of logits, one per categorical feature
        logits_list = self.forward(masked_categorical, masked_continuous)

        # Compute loss only on masked positions for each feature
        # logits_list: list of [batch, num_classes_i] tensors
        # original_categorical: [batch, num_features]
        # mask: [batch, num_features]

        total_loss = 0
        num_masked_total = 0

        # Compute loss per feature, only on masked positions
        for feature_idx, logits in enumerate(logits_list):
            # Get mask for this feature
            feature_mask = mask[:, feature_idx]
            num_masked = feature_mask.sum()

            if num_masked > 0:
                # Select only masked samples for this feature
                masked_logits = logits[feature_mask]
                masked_targets = original_categorical[:, feature_idx][feature_mask]

                # Filter out any invalid targets (e.g., -1 or >= num_classes)
                num_classes = logits.shape[1]
                valid_mask = (masked_targets >= 0) & (masked_targets < num_classes)

                if valid_mask.sum() > 0:
                    masked_logits = masked_logits[valid_mask]
                    masked_targets = masked_targets[valid_mask]

                    # Compute cross-entropy loss for this feature
                    feature_loss = self.loss(masked_logits, masked_targets)
                    total_loss += feature_loss
                    num_masked_total += valid_mask.sum()

        # Average loss
        if num_masked_total > 0:
            total_loss = total_loss / len(logits_list)

        # Log the training loss
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original_categorical = batch["original_categorical"]
        mask = batch["mask"]

        # Forward pass
        logits_list = self.forward(masked_categorical, masked_continuous)

        # Compute loss only on masked positions for each feature
        total_loss = 0
        num_masked_total = 0

        for feature_idx, logits in enumerate(logits_list):
            feature_mask = mask[:, feature_idx]
            num_masked = feature_mask.sum()

            if num_masked > 0:
                masked_logits = logits[feature_mask]
                masked_targets = original_categorical[:, feature_idx][feature_mask]

                # Filter out any invalid targets
                num_classes = logits.shape[1]
                valid_mask = (masked_targets >= 0) & (masked_targets < num_classes)

                if valid_mask.sum() > 0:
                    masked_logits = masked_logits[valid_mask]
                    masked_targets = masked_targets[valid_mask]
                    feature_loss = self.loss(masked_logits, masked_targets)
                    total_loss += feature_loss
                    num_masked_total += valid_mask.sum()

        if num_masked_total > 0:
            total_loss = total_loss / len(logits_list)

        # Log the validation loss
        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original_categorical = batch["original_categorical"]
        mask = batch["mask"]

        # Forward pass
        logits_list = self.forward(masked_categorical, masked_continuous)

        # Compute loss only on masked positions for each feature
        total_loss = 0
        num_masked_total = 0

        for feature_idx, logits in enumerate(logits_list):
            feature_mask = mask[:, feature_idx]
            num_masked = feature_mask.sum()

            if num_masked > 0:
                masked_logits = logits[feature_mask]
                masked_targets = original_categorical[:, feature_idx][feature_mask]

                # Filter out any invalid targets
                num_classes = logits.shape[1]
                valid_mask = (masked_targets >= 0) & (masked_targets < num_classes)

                if valid_mask.sum() > 0:
                    masked_logits = masked_logits[valid_mask]
                    masked_targets = masked_targets[valid_mask]
                    feature_loss = self.loss(masked_logits, masked_targets)
                    total_loss += feature_loss
                    num_masked_total += valid_mask.sum()

        if num_masked_total > 0:
            total_loss = total_loss / len(logits_list)

        # Log the test loss
        self.log("test_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
