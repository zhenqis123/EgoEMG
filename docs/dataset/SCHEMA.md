# EgoEMG field reference

## Paths and arrays

Follow `dataset.json.sessions[].path` to the session manifest. Each array
specifies `path`, `dtype`, `shape` and usually `time_axis`. `.dat` arrays are
C-order binary arrays with no header; use the complete declared NumPy dtype
and shape. `.npy` files retain their NumPy header. Session array/video/crop/depth
paths are session-relative. Calibration, object, supplement and policy paths starting with their named root
directories are dataset-relative. Camera geometry is included from V1 onward.

Read a field only when its specification is present. A missing or false-valid
field is unavailable and must be excluded from measurements.

## Hand arrays

Each non-null `hands.left`/`hands.right` has its own row count and time axis.
Never assume left row i and right row i have identical timestamps.

| Field | Typical shape | Units / interpretation |
|---|---|---|
| `timestamps` | N | UTC microseconds on the published hand timeline |
| `source_timestamps`, `receive_timestamps` | N | Optional acquisition clocks; see per-session clock metadata |
| `segment_id`, `continuity_segment`, `recording_index` | N | Optional boundaries; windows must respect changes |
| `emg_raw`, `emg_filtered` | N × 8 | mV; channel order already applied |
| `emg_valid` | N | EMG availability |
| `imu`, `imu_valid` | declared shape | Layout and units in `imu_layout` / `imu_semantics` |
| `keypoints_world` | N × 21 × 3 | metres in the session world frame |
| `keypoints_valid` | N × 21 | Per-point validity |
| `joint_angles` | N × 20 | Finger joint angles, radians |
| `joint_angles_valid` or `label_valid` | N | Finger-target validity; prefer the former when present |
| `wrist_angles` | N × 2 | Pitch then yaw, degrees in storage |
| `wrist_valid` | N | Wrist-target validity |
| `mano_pose` | N × 48 | Axis-angle radians, canonical MANO-right semantics |
| `mano_beta` | 1 × 10, N × 10, or R × 10 | Static, per-hand-row, or per-recording shape parameters |
| `split_id` | N | Generalization category; combine with the evaluation protocol |
| `gesture_class`, `gesture_active` | N | Root-vocabulary index and activity flag |
| `object_id` | N | Optional canonical object number; negative values mean unavailable |

`emg_channel_order.output_to_source[j]` records the source channel represented
by output channel j. When `applied` is true, do not permute again. Filter
parameters and implementation are in `hands.<side>.filter`; read the provided
filtered samples directly or use raw EMG for another preprocessing choice.

Finger-angle order is:

```text
 0 THUMB_CMC_FE     1 THUMB_CMC_AA     2 THUMB_MCP_FE    3 THUMB_IP_FE
 4 INDEX_MCP_AA     5 INDEX_MCP_FE     6 INDEX_PIP_FE    7 INDEX_DIP_FE
 8 MIDDLE_MCP_AA    9 MIDDLE_MCP_FE   10 MIDDLE_PIP_FE  11 MIDDLE_DIP_FE
12 RING_MCP_AA     13 RING_MCP_FE     14 RING_PIP_FE    15 RING_DIP_FE
16 PINKY_MCP_AA    17 PINKY_MCP_FE    18 PINKY_PIP_FE   19 PINKY_DIP_FE
```

FE denotes flexion/extension and AA denotes abduction/adduction. The standalone
`angles()` reader appends wrist pitch/yaw as columns 20/21, converts wrist
values to radians, and returns a per-angle mask. Finger validity gates all
22 targets, and `wrist_valid` additionally gates the final two. Nonfinite
values are zero-filled and masked. Do not count masked values in errors.

For beta arrays with `time_axis: recording`, locate the half-open
`hands.<side>.recordings[].samples = [start,end)` interval and use its
`beta_index`. Never index this array directly with a hand sample row.
Both hand pose arrays use MANO-right semantics. Recover left-hand geometry
with the dataset's left reflection/alignment convention. Native motion-capture fits under `native_generated_labels`
have their own timestamp axis and must not be indexed as hand sample rows.

## Time alignment

Time is in microseconds unless a field declares a different unit. Original,
historical ShowEE and recent captures have different clock provenance; the
per-stream `timing` and per-hand clock metadata are authoritative. Do not
replace timestamp information with a nominal frame-rate calculation.

Historical ShowEE hand arrays preserve some clock reversals. For timestamp
lookup, stably sort timestamps and map results back to original rows. For EMG
windows, split at clock reversals, gaps greater than 100 ms, recording/segment
changes, invalid EMG spans and split/role changes. Do not reorder the stored
EMG array or build windows across these boundaries. The training repository
implements these window rules; the standalone reader returns requested rows.

Camera alignment in the training reader uses the preceding hand sample with
an age at most 50 ms. Apply validity masks to the aligned samples.

## Cameras, calibration and RGB

`cameras.<device>.streams.<stream>.arrays` describes logical camera rows.
Typical fields are `timestamps`, `timestamp_valid`, `world_poses`, and
`pose_valid`. `world_poses` maps camera-local points into the session world
frame. If `world_pose_storage == static`, its one 4×4 matrix is shared by all
rows; otherwise index per row. Use pose and timestamp masks when provided.

The video can contain more physical frames than the logical timestamp array.
If `video.frame_index_map` exists, translate logical row i through that array
before decoding. Otherwise the logical row is the physical ordinal in the
published video. `selected_source_frames` can refer to the pre-export source. Apply only `frame_index_map` to the published video.

Use `session.rgb(camera, stream, row)` for this mapping plus decord decoding.
For repeated reads, reuse a decord VideoReader and call `physical_frame()`.

`intrinsics_path` and `extrinsics_path` are dataset-relative. Read K, image size,
distortion model and distortion coefficients from the selected calibration;
do not assume a pinhole or shared distortion model across devices. Intrinsics
apply to the declared pixel domain. Per-session camera-world poses and fixed
sensor-to-sensor extrinsics describe different transforms.

## Hand crops

Use the variant matching `hand_crops.active_id`. `valid[row,hand]` indicates
availability. `offsets[row,hand]` contains byte offset and length within
`storage.path`; the bytes decode to one JPEG. `hand_order`, `horizontal_flip`,
`bbox_xyxy`, and `source_to_crop` specify hand ordering and geometry. The reader
returns an RGB array or None for an unavailable crop.

## Depth (V2 and later)

V1 contains no depth payload or metadata. V2 adds depth values, using the
published calibration and the declared depth/RGB coordinate conventions.

Depth-container rows form a separate sampled timeline. `depth.h5` contains
`/depth` as uint16 `[D,H,W]` and `/timestamps` as actual UTC microseconds.
Multiply by `depth.scale_to_meters`; exclude `invalid_pixel_values`.
`session.depth(camera,row)` takes a depth-container row and returns metres,
pixel validity and timestamp. Also consult the declared frame/timestamp masks
when selecting depth rows for an application.

`native_frame_index`, where declared, maps the logical depth metadata rows to
the published container rows. `/paired_rgb_frame_index`, when present, retains
its declared RGB pairing. Depth row i is not generally RGB row i. Depth is
sampled at source logical rows 0,3,6,...; invalid candidates were dropped
without replacement. Use the recorded timestamps for alignment.

The lossless filter chain is RVL (40517), Fletcher32 and Zstd (32015). Load the
supplied RVL plugin and `hdf5plugin`; see `readers/depth_rvl/README.md` in V2 and later.

## Object masks and meshes

The supplement manifest lists available SAM3 Parquet files. Its `frame_idx`
indexes the logical head-camera timestamp array, and `ts_us` must match that
row. Use the video mapping separately when displaying a mask on decoded RGB.
`readers/sam3_masks.py` reconstructs full-resolution boolean masks from cropped
packed payloads. Missing annotations indicate unknown visibility.

Object IDs, names and bindings are defined by `objects/index.json`
(`objects` for recording identities; `categories` for canonical categories).
`objects/mesh_manifest.json` holds the sole asset registry, file checksums,
asset aliases, dimension comparisons and
texture dependencies; coordinates are metres. The mesh frame has not been
registered as an object pose trajectory. Use only explicitly bound assets.

Recording-completeness reports are combined in `sessions/object_coverage.json`,
keyed by session ID under `sessions`. A session's `object_coverage.path` is
relative to its session folder; `session_key` selects the combined report.

## Object poses (V3)

V1 and V2 contain no object pose trajectories. V3 requires session-level
`object_poses` entries declaring `object_id`, `asset_id`, `frame`, `units: m`,
`transform_convention: asset_to_world` and `arrays` containing `timestamps`,
`transforms` and `valid`. Transforms are `[N,4,4]` matrices mapping asset-frame points in metres
into the declared session frame; timestamps are UTC microseconds, and `valid`
is a per-row mask. Asset IDs bind to `objects/mesh_manifest.json`. Invalid rows
are unavailable. Refer to the dataset version for object-pose availability.
