# EgoEMG dataset viewer

Browse synchronized video, EMG signals, hand geometry and available annotations.
The viewer reads the session dataset directly; basic video and signal browsing
requires no model checkpoints or MANO assets.

## Launch

For a first run, use the [20-minute demo subset](../../../docs/dataset/README.md#demo-subset).
Pass its directory with `--dataset-dir`.

Complete the [repository setup](../../../README.md#setup), then run from the
repository root:

```bash
python scripts/tools/dataset_viewer/launch.py \
  --dataset-dir "$EGOEMG_ROOT/data/EgoEMG_dataset" --port 8793
```

Open `http://127.0.0.1:8793/` on the machine running the viewer. For a server,
follow [Remote access](#remote-access).

## Browse sessions

Select a session to inspect its synchronized video, signals and available hand
annotations. The session library is ordered by recording date, oldest first.
Each date is grouped into Gesture and HOI (hand–object interaction), with
recordings ordered by start time within each task. Search supports dates,
participants, session IDs and task names.

The viewer displays published annotations and their validity masks, including
available native hand fits. Depth views are available with V2 or later data. The launcher refreshes when `dataset.json` changes.

## Optional MANO rendering

MANO mesh rendering requires licensed MANO model assets and `manotorch`.
Install `manotorch` from your separately obtained source checkout:

```bash
python -m pip install -e /path/to/manotorch
mkdir -p "$EGOEMG_ROOT/pretrained_models"
ln -s /path/to/mano_v1_2 "$EGOEMG_ROOT/pretrained_models/mano_assets"
```

The asset directory must contain `models/MANO_RIGHT.pkl`. Set
`MANO_ASSETS_ROOT` to use another location. If `EGOEMG_ROOT` is unset, the
viewer looks under the repository's `pretrained_models/mano_assets/`.

## Remote access

The viewer listens on `127.0.0.1` on the machine running it. In a VS Code
Remote-SSH window, open **Ports**, choose **Forward a Port**, enter `8793`,
and open the displayed **Local Address**. The local port may differ if `8793`
is already in use on your computer.

Alternatively, run this command on your local machine:

```bash
ssh -N -L 127.0.0.1:8793:127.0.0.1:8793 your-server
```

Then open `http://127.0.0.1:8793/`. To run another instance, choose another
port and forward that same port:

```bash
python scripts/tools/dataset_viewer/launch.py \
  --dataset-dir "$EGOEMG_ROOT/data/EgoEMG_dataset" --port 8794
```

## Logs and depth support

Logs and status files are stored under `logs/tools/dataset_viewer/` relative
to the repository root. The default instance uses `server.log` and
`process.json`; custom ports use `server_<port>.log`, `process_<port>.json`
and separate lock files.

For RVL-compressed depth, set `COLLECTOR_RVL_PLUGIN_DIR` to the directory
containing your built HDF5 plugin. The launcher preserves this setting and
reuses RVL filter binaries from `scripts/tools/depth_rvl/native/lib`.
See `readers/depth_rvl/README.md` in the downloaded V2-or-later dataset for
native reader build instructions.
