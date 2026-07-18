from pathlib import Path

import pytorch_lightning as pl


class LocalFileModelCheckpoint(pl.callbacks.ModelCheckpoint):
    """ModelCheckpoint variant that avoids DDP object broadcast for file checks."""

    def file_exists(self, filepath: str, trainer: pl.Trainer) -> bool:
        return Path(filepath).exists()
