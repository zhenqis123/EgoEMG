# A Note From the EMG2Pose Developers

For forward kinematics (converting joint angles into 3D hand landmark positions), we
leverage the [UmeTrack](https://github.com/facebookresearch/umetrack) library.

However, this package is not a Python package (no `setup.py` file). As such,
for the initial code submission, we opt to simply copy in the source files
(under `emg2pose.UmeTrack`) so that we can import and use the forward kinematics utilities.

For the camera-ready version on GitHub, we will seek a more robust solution -- for example
using GitHub's submodule feature to symlink the repository and/or make a PR to `UmeTrack`
to add the necessary `setup.py` and `__init__.py` files directly to the repo itself.

For now, we make a few minor modifications to the `UmeTrack` source code in order to make it work
as a package.

1) We add blank `__init__.py` files to make submodules importable
2) We opt to explicitly include a pared-down version of the `pytorch3d.transforms.so3` source
file for the single dependency on the `so3_exp_map` function. We opt for this choice as to
avoid the complicated builds necessary to add the full `pytorch3d` library as a dependency.

# References / Credit

* [UmeTrack](https://github.com/facebookresearch/umetrack)
* [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
