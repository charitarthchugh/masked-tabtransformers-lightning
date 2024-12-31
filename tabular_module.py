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
        null_token: int = 1,
        return_attention: bool = False,
        leasing_rate: float = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if not mlp_act:
            mlp_act = nn.ReLU()
        self.null_token = null_token
        self.return_attention = return_attention
        self.learning_rate = leasing_rate
        # model instantiation copied from example
        self.model = TabTransformer(
            categories=categorical_unique_counts,
            num_continuous=num_continuous,  # number of continuous values
            dim=dim,  # dimension, paper set at 32
            dim_out=len(categorical_unique_counts) + num_continuous,  # size of input
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
        print(self.model)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original = batch["original"]
        outputs = self.model(
            masked_categorical, masked_continuous, self.return_attention
        )
        # Compute the loss
        loss = self.loss(outputs, original)

        # Log the training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original = batch["original"]
        outputs = self.model(
            masked_categorical, masked_continuous, self.return_attention
        )
        # Compute the loss
        loss = self.loss(outputs, original)

        # Log the validation loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx) -> STEP_OUTPUT:
        # Extract data from the batch
        masked_categorical = batch["masked_categorical"]
        masked_continuous = batch["masked_continuous"]
        original = batch["original"]
        outputs = self.model(
            masked_categorical, masked_continuous, self.return_attention
        )
        # Compute the loss
        loss = self.loss(outputs, original)

        # Log the test loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
