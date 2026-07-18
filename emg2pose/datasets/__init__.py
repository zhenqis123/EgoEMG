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
    "EgoEmgIncreDataset",
]

if TYPE_CHECKING:
    from emg2pose.datasets.emg2pose_dataset import Emg2PoseDataset
    from emg2pose.datasets.emg2pose_dataset_legacy import (
        Emg2PoseSessionData,
        WindowedEmgDataset,
    )
    from emg2pose.datasets.multisession_emg2pose_dataset_legacy import (
        MultiSessionWindowedEmgDataset,
    )
    from emg2pose.datasets.emg2qwerty_dataset import Emg2QwertyDataset
    from emg2pose.datasets.pimforce_dataset import (
        PimforceDataset,
        PimforceSessionDataset,
    )
    from emg2pose.datasets.ninapro_dataset import (
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
    from emg2pose.datasets.posture_dataset import PostureDataset
    from emg2pose.datasets.grabmyo_dataset import GrabMyoDataset
    from emg2pose.datasets.putemg_dataset import PutEmgDataset
    from emg2pose.datasets.myoki_dataset import MyoKiDataset
    from emg2pose.datasets.pretrain_wrapper import PretrainWrapperDataset
    from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset


def __getattr__(name: str):
    if name == "Emg2PoseDataset":
        from emg2pose.datasets.emg2pose_dataset import Emg2PoseDataset

        return Emg2PoseDataset
    if name in {"Emg2PoseSessionData", "WindowedEmgDataset"}:
        from emg2pose.datasets.emg2pose_dataset_legacy import (
            Emg2PoseSessionData,
            WindowedEmgDataset,
        )

        return Emg2PoseSessionData if name == "Emg2PoseSessionData" else WindowedEmgDataset
    if name == "MultiSessionWindowedEmgDataset":
        from emg2pose.datasets.multisession_emg2pose_dataset_legacy import (
            MultiSessionWindowedEmgDataset,
        )

        return MultiSessionWindowedEmgDataset
    if name == "Emg2QwertyDataset":
        from emg2pose.datasets.emg2qwerty_dataset import Emg2QwertyDataset

        return Emg2QwertyDataset
    if name in {"PimforceDataset", "PimforceSessionDataset"}:
        from emg2pose.datasets.pimforce_dataset import (
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
        from emg2pose.datasets import ninapro_dataset as _ninapro

        return getattr(_ninapro, name)
    if name == "PostureDataset":
        from emg2pose.datasets.posture_dataset import PostureDataset

        return PostureDataset
    if name == "GrabMyoDataset":
        from emg2pose.datasets.grabmyo_dataset import GrabMyoDataset

        return GrabMyoDataset
    if name == "PutEmgDataset":
        from emg2pose.datasets.putemg_dataset import PutEmgDataset

        return PutEmgDataset
    if name == "MyoKiDataset":
        from emg2pose.datasets.myoki_dataset import MyoKiDataset

        return MyoKiDataset
    if name == "PretrainWrapperDataset":
        from emg2pose.datasets.pretrain_wrapper import PretrainWrapperDataset

        return PretrainWrapperDataset
    if name == "EgoEmgMemmapDataset":
        from emg2pose.datasets.egoemg_memmap_dataset import EgoEmgMemmapDataset

        return EgoEmgMemmapDataset
    if name == "EgoEmgVisionDataset":
        from emg2pose.datasets.egoemg_vision_dataset import EgoEmgVisionDataset

        return EgoEmgVisionDataset
    if name == "EgoEmgIncreDataset":
        from emg2pose.datasets.egoemg_incre_dataset import EgoEmgIncreDataset

        return EgoEmgIncreDataset
    raise AttributeError(name)
