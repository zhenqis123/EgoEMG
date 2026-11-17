"""SensingDynamics direct angle-regression module."""

from torch import nn

from egoemg.models.modules.base import BaseModule


class SensingDynamicsModule(BaseModule):
    """Predict joint angles with the SensingDynamics encoder and MLP.

    Unlike velocity-based emg2pose, this baseline directly predicts angles and
    therefore does not consume the ground-truth initial pose at inference.
    """

    def __init__(
        self,
        featurizer: nn.Module,
        decoder: nn.Module,
        out_channels: int = 22,
    ) -> None:
        super().__init__(
            featurizer=featurizer,
            decoder=decoder,
            out_channels=out_channels,
            provide_initial_pos=False,
        )
