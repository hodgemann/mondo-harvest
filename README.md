# mondo-harvest

[![tests](https://github.com/hodgemann/mondo-harvest/actions/workflows/tests.yml/badge.svg)](https://github.com/hodgemann/mondo-harvest/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Harvest the prompts left inside AI-generated images and videos, and merge them into
[Mondocraft](#merging-into-mondocraft)'s saved prompts. The crawl works on its own; the merge works with Mondocraft.

Point it at a folder of outputs, even one with tens of thousands of files, and it lists every distinct prompt once,
with what made each output: model, LoRAs, negative prompt, sampler settings and seed. Only metadata is read; image and
video data are never decoded, and the source files are never changed.

## What it reads

- **ComfyUI** outputs (PNG, WebP, JPEG, and MP4/MOV/WebM/MKV from Video Combine or Save Video). The workflow graph is
  followed from the file's save node back to its sampler, so the harvest holds the prompt that actually made the
  output, not every text box in the workflow. It also records:
  - the model (checkpoint or UNET) and text encoders;
  - the LoRAs switched on, with their strengths (LoRA nodes that are bypassed, switched off or at strength 0 are left
    out), including rgthree's Power Lora Loader and Lora Loader Stack;
  - the LoRAs' trigger words, taken off the start of the prompt (see [LoRA trigger words](#lora-trigger-words));
  - the negative prompt, sampler, scheduler, steps, CFG or guidance, the number of sampling passes and the output size;
  - each output's seed;
  - unused prompt text left in the workflow (a muted or bypassed prompt node, a node still showing the prompt before),
    listed with the files it sat in;
  - whether a reference image fed the generation.

  A video's first-frame PNG saved beside it counts as the same output.
- **Everything else**: prompts that generators and services leave in description, comment or caption fields: PNG
  text chunks, EXIF (ImageDescription, UserComment, XPComment, …), XMP (`dc:description` and others, plus any field
  named for a prompt), IPTC (Caption) and video tags (`description`, `comment`, `prompt`, …). A field counts only when
  its name says it can hold a prompt and its text reads like one, so titles, camera, software, author and copyright
  fields, short labels, links, IDs and credit lines are skipped. A file whose metadata marks it as a photograph or a
  rights-managed scan (rights, author or camera fields) gives no prompts unless it also carries an AI marker (a C2PA
  credential, an AIGC label, or IPTC's "trained algorithmic media" source type): museum and stock-photo captions read
  much like prompts. JSON fields are read for their prompt, negative prompt, seed and model keys.
  `Prompt / Negative prompt: … / Steps: …, Seed: …` text is split into its parts. Every prompt records the field it
  came from.

Each crawl also counts every metadata field it met, by name only, so you can see where a folder's files keep their
information, including fields it did not take a prompt from.

## Requirements

Python 3.10 or later, with no other packages. Reading videos needs `ffprobe` (part of FFmpeg) on the PATH. Merging
needs a Mondocraft installation (see below).

## Install

mondo-harvest is a single file. Clone the repository:

    git clone https://github.com/hodgemann/mondo-harvest.git
    cd mondo-harvest

or download `mondo_harvest.py` from the [latest release](https://github.com/hodgemann/mondo-harvest/releases/latest).
`python3 mondo_harvest.py --version` prints the version.

## Crawl

    python3 mondo_harvest.py crawl /path/to/outputs [/path/to/more ...]

Each folder named gets one harvest file, `harvests/<folder name>.json`, with its subfolders included
(`--no-recurse` reads only the folder itself; `--out DIR` writes the harvest files elsewhere). Crawling a folder again
replaces its file.

The crawl prints its progress every 500 files and, at the end, the totals, the metadata fields it met (how many files
had each, and how many of those gave a prompt) and the merge commands to copy.

A long crawl saves its progress every two minutes, and when you press Ctrl+C, to `harvests/<folder name>.json.partial`.
If the crawl is cut short, run the same command again: it carries on from there and reads again any files that
failed. If a file it already read has changed since, it starts over.

### LoRA trigger words

A LoRA's trigger word only means something while that LoRA is loaded; saved without it, the trigger pollutes the
prompt. So when a ComfyUI prompt starts with a trigger of one of the LoRAs switched on for that output, the crawl takes
it off the prompt and keeps it with the LoRA instead (`"trigger"` in the setup's LoRA entry). `ohwx woman, a portrait
in the rain` made with `ohwx_woman_v2.safetensors` is harvested as `a portrait in the rain`, and is the same prompt as
`a portrait in the rain` made without the trigger.

A trigger is recognised by the LoRA's file name: a comma-separated piece at the start of the prompt, of up to four
words, that spells consecutive words of the name, separators and case aside (`Mjolnir style` for
`Mjolnir_Style-FLUX`, `pixelart` for `pixel-art-xl-v1.1`, `(ohwx:1.2)` for `OhwxMan_e10`). Name words that only say
what the file is (versions, base models such as `flux` or `sdxl`, `lora`, `style`) never count on their own, and
nothing is taken off unless what is left still reads as a prompt. Triggers that are not in the LoRA's name, or that
sit later in the prompt, stay.

The crawl lists every trigger it took off, with the LoRA and the number of prompts, and the harvest file keeps the
list under `lora_triggers`, so a name word taken off by mistake is easy to spot. `--keep-triggers` leaves prompts as
they are.

### The harvest file

    {
     "format": "mondo-harvest/1",
     "folder": "/path/to/outputs",
     "files_scanned": 117, "files_with_prompts": 117, "prompts_found": 30,
     "metadata_fields": {"png:prompt": {"files": 117, "prompts": 117}, "png:workflow": {"files": 117, "prompts": 0}},
     "lora_triggers": [{"lora": "ohwx_woman_v2.safetensors", "trigger": "ohwx woman", "prompts": 12}],
     "files_without_prompts": [], "errors": [],
     "prompts": [
      {"text": "…", "kind": "image", "t2i": true, "first_seen": "2025-11-03", "last_seen": "2025-11-03", "images": 4,
       "setups": [
        {"source": "comfyui", "models": ["…"], "text_encoders": ["…"],
         "loras": [{"name": "…", "strength": 0.93, "trigger": "…"}], "negative": "…", "sampler": "euler", "scheduler": "simple",
         "steps": 50, "cfg": 2.5, "denoise": 1.0, "width": 1088, "height": 1920,
         "runs": [{"file": "ComfyUI_00006_.png", "date": "2025-11-03", "seed": 66654445561025}, …]}]}]
    }

A prompt is an image or a video prompt by its output. `t2i` (ComfyUI only) is false when a reference image fed the
generation. A prompt used with different settings has one setup for each; `unused_in` lists the files whose workflow
held the text without using it.

## Merging into Mondocraft

Mondocraft, an AI image and video generation app (its GitHub release is coming), keeps saved prompts per workspace. `merge` adds harvested prompts to one:

    python3 mondo_harvest.py merge harvests/outputs.json --dry-run     # counts only, writes nothing
    python3 mondo_harvest.py merge harvests/outputs.json               # into the Main workspace
    python3 mondo_harvest.py merge harvests/*.json --workspace Test
    python3 mondo_harvest.py workspaces                               # lists the workspaces

`--mondocraft DIR` names your Mondocraft folder (default: `~/mondocraft`, or the `MONDOCRAFT` environment variable).
Run merge with Mondocraft's own Python (`<mondocraft>/.venv/bin/python`), which has the packages Mondocraft needs; the
commands a crawl prints already do. Merge uses Mondocraft's own code to read and write the saved prompts, so an
encrypted prompt file stays encrypted.

- New prompts get the dates of their first and last output, so they sort among the prompts you used at that time.
- A prompt that is already saved (the same text apart from white space, for the same kind) is not added again; the
  harvested data is combined with what is there. Merging the same harvest twice changes nothing.
- The harvested data goes under each prompt's `harvest` key: the folders it came from, and its setups with their
  seeds.

Merge while Mondocraft is stopped, or reload its page afterwards.

## Tests

    python3 -m unittest discover -s tests

The tests build small synthetic outputs in a temporary folder and merge into a stand-in for Mondocraft.

## Changes

See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT; see [`LICENSE`](LICENSE).
