# Changelog

All notable changes to mondo-harvest are recorded here. Versions follow [Semantic Versioning](https://semver.org/).

## [1.1.1] - 2026-11-11

### Changed

- The README and `--help` describe the harvest file as the result; merging into Mondocraft is an optional second step.
- The crawl prints the merge commands only when Mondocraft is installed (in `~/mondocraft`, or the folder the
  `MONDOCRAFT` environment variable names). Nothing is searched for.

## [1.1.0] - 2026-11-11

### Added

- LoRA trigger words at the start of a ComfyUI prompt are taken off and kept with the LoRA (`"trigger"`), when they
  spell part of the LoRA's file name. The crawl lists them, and the harvest file keeps them under `lora_triggers`.
  `--keep-triggers` turns this off.

### Changed

- Files whose metadata marks them as photographs or rights-managed scans (camera, author, copyright or rights fields)
  give no prompts unless they also carry an AI marker (C2PA, an AIGC label, or IPTC's algorithmic-media source type).
- Title and headline fields are no longer read as prompts. XMP fields of unknown namespaces count only when their name
  says "prompt", and XMP namespace prefixes skip version numbers in the namespace URI.

## [1.0.0] - 2026-11-11

First public release.

- `crawl`: harvest the prompts from a folder of AI outputs' metadata into one harvest file, with resumable progress.
  - ComfyUI PNG, WebP, JPEG and video outputs: the prompt that fed the sampler, followed through the workflow graph,
    with models, text encoders, active LoRAs and strengths, negative prompt, sampler settings, output size and seeds;
    unused prompt text left in the workflow; whether a reference image fed the generation.
  - Other generators and services: prompts in PNG text, EXIF, XMP, IPTC and video tags, including
    `Prompt / Negative prompt: … / Steps: …` text and JSON fields.
  - A count of every metadata field met, by name.
- `merge`: add harvested prompts to a Mondocraft workspace's saved prompts, without duplicates; `--dry-run` counts only.
- `workspaces`: list Mondocraft's workspaces.
- `--version`.

[1.1.1]: https://github.com/hodgemann/mondo-harvest/releases/tag/v1.1.1
[1.1.0]: https://github.com/hodgemann/mondo-harvest/releases/tag/v1.1.0
[1.0.0]: https://github.com/hodgemann/mondo-harvest/releases/tag/v1.0.0
