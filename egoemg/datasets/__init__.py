from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "Emg2PoseDataset",
    "Emg2PoseSessionData",
    "WindowedEmgDataset",
    "MultiSessionWindowedEmgDataset",
    "Emg2QwertyDataset",
    "PimforceDataset",
    "PimforceSessionDataset",
    "NinaproDataset",
    "NinaproDB1Dataset",
    "NinaproDB2Dataset",
    "NinaproDB3Dataset",
    "NinaproDB4Dataset",
    "NinaproDB5Dataset",
    "NinaproDB6Dataset",
    "NinaproDB7Dataset",
    "NinaproDB8Dataset",
    "PostureDataset",
    "GrabMyoDataset",
    "PutEmgDataset",
    "MyoKiDataset",
    "PretrainWrapperDataset",
    "EgoEmgMemmapDataset",
    "EgoEmgVisionDataset",
    "NinaproMemmapDataset",
    "EgoEmgIncreDataset",
]

if TYPE_CHECKING:
    from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset
    from egoemg.datasets.emg2pose_dataset_legacy import (
        Emg2PoseSessionData,
        WindowedEmgDataset,
    )
    from egoemg.datasets.multisession_emg2pose_dataset_legacy import (
        MultiSessionWindowedEmgDataset,
    )
    from egoemg.datasets.emg2qwerty_dataset import Emg2QwertyDataset
    from egoemg.datasets.pimforce_dataset import (
        PimforceDataset,
        PimforceSessionDataset,
    )
    from egoemg.datasets.ninapro_dataset import (
        NinaproDataset,
        NinaproDB1Dataset,
        NinaproDB2Dataset,
        NinaproDB3Dataset,
        NinaproDB4Dataset,
        NinaproDB5Dataset,
        NinaproDB6Dataset,
        NinaproDB7Dataset,
        NinaproDB8Dataset,
    )
    from egoemg.datasets.posture_dataset import PostureDataset
    from egoemg.datasets.grabmyo_dataset import GrabMyoDataset
    from egoemg.datasets.putemg_dataset import PutEmgDataset
    from egoemg.datasets.myoki_dataset import MyoKiDataset
    from egoemg.datasets.pretrain_wrapper import PretrainWrapperDataset
    from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset


def __getattr__(name: str):
    if name == "Emg2PoseDataset":
        from egoemg.datasets.emg2pose_dataset import Emg2PoseDataset

        return Emg2PoseDataset
    if name in {"Emg2PoseSessionData", "WindowedEmgDataset"}:
        from egoemg.datasets.emg2pose_dataset_legacy import (
            Emg2PoseSessionData,
            WindowedEmgDataset,
        )

        return Emg2PoseSessionData if name == "Emg2PoseSessionData" else WindowedEmgDataset
    if name == "MultiSessionWindowedEmgDataset":
        from egoemg.datasets.multisession_emg2pose_dataset_legacy import (
            MultiSessionWindowedEmgDataset,
        )

        return MultiSessionWindowedEmgDataset
    if name == "Emg2QwertyDataset":
        from egoemg.datasets.emg2qwerty_dataset import Emg2QwertyDataset

        return Emg2QwertyDataset
    if name in {"PimforceDataset", "PimforceSessionDataset"}:
        from egoemg.datasets.pimforce_dataset import (
            PimforceDataset,
            PimforceSessionDataset,
        )

        return PimforceDataset if name == "PimforceDataset" else PimforceSessionDataset
    if name in {
        "NinaproDataset",
        "NinaproDB1Dataset",
        "NinaproDB2Dataset",
        "NinaproDB3Dataset",
        "NinaproDB4Dataset",
        "NinaproDB5Dataset",
        "NinaproDB6Dataset",
        "NinaproDB7Dataset",
        "NinaproDB8Dataset",
    }:
        from egoemg.datasets import ninapro_dataset as _ninapro

        return getattr(_ninapro, name)
    if name == "PostureDataset":
        from egoemg.datasets.posture_dataset import PostureDataset

        return PostureDataset
    if name == "GrabMyoDataset":
        from egoemg.datasets.grabmyo_dataset import GrabMyoDataset

        return GrabMyoDataset
    if name == "PutEmgDataset":
        from egoemg.datasets.putemg_dataset import PutEmgDataset

        return PutEmgDataset
    if name == "MyoKiDataset":
        from egoemg.datasets.myoki_dataset import MyoKiDataset

        return MyoKiDataset
    if name == "PretrainWrapperDataset":
        from egoemg.datasets.pretrain_wrapper import PretrainWrapperDataset

        return PretrainWrapperDataset
    if name == "EgoEmgMemmapDataset":
        from egoemg.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

        return EgoEmgMemmapDataset
    if name == "EgoEmgVisionDataset":
        from egoemg.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

        return EgoEmgVisionDataset
    if name == "EgoEmgIncreDataset":
        from egoemg.datasets.egoemg_incre_dataset import EgoEmgIncreDataset

        return EgoEmgIncreDataset
    raise AttributeError(name)
