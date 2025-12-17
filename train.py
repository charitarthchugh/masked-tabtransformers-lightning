import click
import pandas as pd
import pytorch_lightning as pl
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from tabular_datamodule import TabularDataModule
from tabular_module import TabTransformerModuleforMLM


@click.command()
@click.option(
    "--train-data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the training dataset (CSV format).",
)
@click.option(
    "--val-data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the validation dataset (CSV format).",
)
@click.option(
    "--test-data-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the test dataset (CSV format).",
)
@click.option(
    "--categorical-columns",
    type=str,
    required=True,
    help="Comma-separated names of categorical columns.",
)
@click.option(
    "--numerical-columns",
    type=str,
    default="",
    help="Comma-separated names of numerical columns (optional).",
)
@click.option(
    "--batch-size",
    type=int,
    default=128,
    show_default=True,
    help="Batch size for training and evaluation. if input is 'auto', it will be automatically using a batch size "
    " finder",
)
@click.option(
    "--num-epochs",
    type=int,
    default=10,
    show_default=True,
    help="Number of training epochs.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.001,
    show_default=True,
    help="Learning rate for the optimizer.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="./outputs",
    show_default=True,
    help="Directory to save model checkpoints and logs.",
)
@click.option(
    "--logger",
    type=click.Choice(["tensorboard", "wandb"], case_sensitive=False),
    default="wandb",
)
@click.option(
    "--wandb-project-name",
    type=str,
    default="TabTransformer",
    show_default=True,
    help="Name of the Wandb project to log to.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
def train_script(
    train_data_path,
    val_data_path,
    test_data_path,
    categorical_columns,
    numerical_columns,
    batch_size,
    num_epochs,
    learning_rate,
    output_dir,
    logger,
    wandb_project_name,
    seed,
):
    """
    Training script for TabTransformer using Click for CLI.
    It initializes the data module, sets up the model, and trains it using PyTorch Lightning.
    """
    # Load datasets
    train_df = pd.read_csv(train_data_path)
    val_df = pd.read_csv(val_data_path)
    test_df = pd.read_csv(test_data_path)

    # Prepare column lists
    categorical_columns = categorical_columns.split(",")
    numerical_columns = numerical_columns.split(",") if numerical_columns else []

    # Initialize DataModule
    data_module = TabularDataModule(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        batch_size=batch_size,
    )

    # Initialize the model
    model = TabTransformerModuleforMLM(
        categorical_unique_counts=data_module.metadata.categorical_cardinality,
        num_continuous=len(data_module.metadata.numerical_col_names),
        dim=32,  # You may make this a CLI option if needed
        depth=6,  # Adjust to your needs
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        null_token=data_module.metadata.categorical_encoder.null_token,
        continuous_mean_std=data_module.continuous_mean_std,
        leasing_rate=learning_rate,
    )
    callbacks = []
    # Setup logger & checkpointing
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=output_dir,
            filename="tab-transformer-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )
    )
    if batch_size == "auto":
        callbacks.append(BatchSizeFinder(mode="binsearch"))
    # Trainer

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=WandbLogger(project=wandb_project_name)
        if logger == "wandb"
        else TensorBoardLogger(
            output_dir,
        ),
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic=True,
        detect_anomaly=True,
    )

    # Start training
    trainer.fit(model, datamodule=data_module)

    # Testing (use the best checkpoint)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    train_script()
