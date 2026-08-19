# Third-party notices

This code pre-release contains or interfaces with third-party components. This
file is a release-facing index, not a substitute for the license text supplied
with each component.

| Component | Repository location / acquisition | License status |
| --- | --- | --- |
| EgoEMG-authored code | Repository files outside identified third-party material | MIT; see root `LICENSE`. |
| UmeTrack code and hand model | `egoemg/UmeTrack/` | CC-BY-NC-4.0; see `egoemg/UmeTrack/LICENSE`. This restriction applies to that material and must be preserved. |
| Meta emg2pose-derived material | Referenced in root `LICENSE` | CC-BY-NC-SA-4.0 attribution/share-alike obligations may apply; verify file-level provenance before redistribution. |
| MANO model files | Not bundled; users must acquire them from the model provider | Separate license; do not redistribute with this repository unless independently authorized. |
| WiLoR weights/assets | Not bundled; configured through external paths in research workflows | Separate upstream terms; not part of this pre-release. |
| PyPI/Conda dependencies | Declared in `setup.py` and `environment.yml` | Each dependency retains its own license. |

The distribution has mixed licensing. Do not treat a package-level metadata field
as permission to relicense all contained material as MIT. A formal data/model
release must complete a file-level provenance and license review before adding
assets or a single unified license assertion.

