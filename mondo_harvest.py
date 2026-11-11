"""mondo-harvest: harvest the prompts left in AI-generated images and videos into a JSON harvest file.

    python3 mondo_harvest.py crawl FOLDER [FOLDER ...] [--out DIR] [--no-recurse]
    python3 mondo_harvest.py merge HARVEST.json [...] [--mondocraft DIR] [--workspace NAME] [--dry-run]
    python3 mondo_harvest.py workspaces [--mondocraft DIR]

crawl reads every output's metadata, never its image or video data, and writes one harvest file per folder named on
the command line, its subfolders included, to harvests/ unless --out says otherwise. Source files are only read.
ComfyUI outputs give the prompt that fed the sampler, any unused prompt text left in the workflow, the models, the
LoRAs switched on, the negative prompt, sampler settings and each output's seed. Other outputs give the prompts found
in their description, comment, caption and similar fields (PNG text, EXIF, XMP, IPTC, video tags). Every distinct
prompt is listed once (runs of white space count as one space), with its setups and, under each, the outputs it
made. The harvest also counts every metadata field it met, to show where a folder's files keep their information.

merge, an optional second step for Mondocraft users, adds a harvest file's prompts to a Mondocraft workspace's saved
prompts, using Mondocraft's own code from its folder, so encryption and workspaces work as in Mondocraft. A prompt
already saved is not added again; merging the same harvest twice changes nothing. crawl needs only Python 3.10+ (and
ffprobe for videos).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path

__version__ = "1.2.0"

APP = Path(__file__).resolve().parent
HARVESTS = APP / "harvests"
FORMAT = "mondo-harvest/1"
OLD_FORMATS = ("mondocraft-comfyui-harvest/1",)        # harvests made before mondo-harvest left Mondocraft
PARTIAL_FORMAT = "mondo-harvest-progress/1"            # never accepted by merge
DEFAULT_MONDOCRAFT = Path(os.environ.get("MONDOCRAFT", "~/mondocraft")).expanduser()

IMAGE_EXT = (".png", ".webp", ".jpg", ".jpeg")
VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv")
MODEL_EXT = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".sft", ".onnx")

# Node inputs that hold prompt text, and node types whose text is documentation rather than a prompt.
TEXT_KEYS = ("text", "text_g", "text_l", "t5xxl", "clip_l", "prompt", "positive", "string", "value", "populated_text",
             "text_positive", "user_prompt")
NOTE_TYPES = re.compile(r"note|markdown|comment|label", re.I)
TEXT_NODE_TYPES = re.compile(r"text|prompt|string|show|display|wildcard|encode|conditioning", re.I)
LOAD_MEDIA = re.compile(r"^(LoadImage|LoadImageMask|LoadImageOutput|LoadVideo|VHS_LoadVideo\w*|VHS_LoadImages\w*)$")
VIDEO_OUTPUT = re.compile(r"VideoCombine|SaveVideo|SaveWEBM|SaveAnimated", re.I)
CHECKPOINT_KEYS = ("ckpt_name", "unet_name", "model_name_unet", "base_ckpt_name")
ENCODER_KEYS = ("clip_name", "clip_name1", "clip_name2", "clip_name3", "clip_name4", "t5_name", "text_encoder")


# ---------------------------------------------------------------- reading metadata (read-only)
#
# read_fields collects every text metadata field of a file under a namespaced name ("png:parameters",
# "exif:ImageDescription", "xmp:dc:description", "iptc:Caption", "jpeg:comment", "video:comment"), without decoding
# any image or video data. C2PA content credentials are noted by name only ("c2pa"), since they hold no prompt text.

EXIF_TAGS = {0x010D: "DocumentName", 0x010E: "ImageDescription", 0x010F: "Make", 0x0110: "Model", 0x0131: "Software",
             0x0132: "DateTime", 0x013B: "Artist", 0x8298: "Copyright", 0x9286: "UserComment", 0x9C9B: "XPTitle",
             0x9C9C: "XPComment", 0x9C9D: "XPAuthor", 0x9C9E: "XPKeywords", 0x9C9F: "XPSubject",
             0xA430: "CameraOwnerName", 0x9003: "DateTimeOriginal"}
IPTC_TAGS = {5: "ObjectName", 25: "Keywords", 80: "Byline", 105: "Headline", 110: "Credit", 115: "Source",
             116: "Copyright", 120: "Caption", 122: "Writer"}
XMP_PREFIX = {"http://purl.org/dc/elements/1.1/": "dc", "http://ns.adobe.com/xap/1.0/": "xmp",
              "http://ns.adobe.com/photoshop/1.0/": "photoshop", "http://ns.adobe.com/exif/1.0/": "exif",
              "http://ns.adobe.com/tiff/1.0/": "tiff", "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/": "Iptc4xmpCore",
              "http://iptc.org/std/Iptc4xmpExt/2008-02-29/": "Iptc4xmpExt", "http://ns.adobe.com/xap/1.0/mm/": "xmpMM",
              "http://ns.adobe.com/xap/1.0/rights/": "xmpRights"}
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def _add(fields: dict, name: str, value: str) -> None:
    value = value.replace("\0", "").strip()
    if name in fields:
        if value and value not in fields[name].split("\n"):
            fields[name] = (fields[name] + "\n" + value).strip()
    else:
        fields[name] = value


def _exif(data: bytes, fields: dict) -> None:
    """Text tags of a TIFF/EXIF block (IFD0 and the EXIF sub-IFD), as exif:<Tag>."""
    if data.startswith(b"Exif\0\0"):
        data = data[6:]
    if data[:2] not in (b"II", b"MM"):
        return
    end = "<" if data[:2] == b"II" else ">"

    def ifd(offset, depth=0):
        if depth > 2 or offset + 2 > len(data):
            return
        (count,) = struct.unpack(end + "H", data[offset:offset + 2])
        for i in range(min(count, 512)):
            e = offset + 2 + 12 * i
            if e + 12 > len(data):
                return
            tag, typ, n = struct.unpack(end + "HHI", data[e:e + 8])
            if tag == 0x8769:                                   # the EXIF sub-IFD
                ifd(struct.unpack(end + "I", data[e + 8:e + 12])[0], depth + 1)
                continue
            if typ not in (1, 2, 7) or n == 0:
                continue
            raw = data[e + 8:e + 8 + n] if n <= 4 else \
                data[(o := struct.unpack(end + "I", data[e + 8:e + 12])[0]):o + n]
            name = EXIF_TAGS.get(tag, f"0x{tag:04X}")
            if name.startswith("XP"):
                text = raw.decode("utf-16-le", "replace")
            elif tag == 0x9286:                                 # UserComment: 8-byte character code, then the text
                code, body = raw[:8], raw[8:]
                text = body.decode("utf-16-be" if end == ">" else "utf-16-le", "replace") \
                    if code.startswith(b"UNICODE") else body.decode("utf-8", "replace")
            elif typ == 2:
                text = raw.decode("utf-8", "replace")
            else:
                continue
            _add(fields, f"exif:{name}", text)

    try:
        ifd(struct.unpack(end + "I", data[4:8])[0])
    except struct.error:
        pass


def _xmp(data: bytes, fields: dict) -> None:
    """XMP properties with text, as xmp:<prefix>:<name> (language alternatives and lists joined by new lines)."""
    import xml.etree.ElementTree as ET
    text = data.decode("utf-8", "replace")
    for first, last in (("<x:xmpmeta", "</x:xmpmeta>"), ("<rdf:RDF", "</rdf:RDF>")):
        start, stop = text.find(first), text.rfind(last)
        if start >= 0 and stop > start:
            break
    else:
        return
    try:
        root = ET.fromstring(text[start:stop + len(last)])
    except ET.ParseError:
        return

    def name(tag):
        uri, _, local = tag[1:].partition("}") if tag.startswith("{") else ("", "", tag)
        prefix = XMP_PREFIX.get(uri) or next((seg for seg in reversed(re.split(r"[/#:]", uri))     # skip versions
                                              if seg and not re.fullmatch(r"[\d.-]+|v\d[\d.]*", seg)), "xmp")
        return f"xmp:{prefix}:{local}"

    parents = {child: parent for parent in root.iter() for child in parent}
    for el in root.iter():
        if el.tag == f"{{{RDF}}}Description":
            for k, v in el.attrib.items():
                if not k.startswith(f"{{{RDF}}}"):
                    _add(fields, name(k), v)
        if el.text and el.text.strip() and not len(el):
            owner = el
            while owner.tag.startswith(f"{{{RDF}}}") and owner in parents:   # rdf:li -> rdf:Alt -> dc:description
                owner = parents[owner]
            if not owner.tag.startswith(f"{{{RDF}}}") and owner.tag != "{adobe:ns:meta/}xmpmeta":
                _add(fields, name(owner.tag), el.text)


def _iptc(data: bytes, fields: dict) -> None:
    """IPTC records inside a Photoshop APP13 block, as iptc:<Name>."""
    i = data.find(b"8BIM\x04\x04")
    if i < 0:
        return
    i += 6
    namelen = data[i]
    i += 1 + namelen + (0 if namelen % 2 else 1)
    (size,) = struct.unpack(">I", data[i:i + 4])
    block, j = data[i + 4:i + 4 + size], 0
    while j + 5 <= len(block) and block[j] == 0x1C:
        rec, ds, n = block[j + 1], block[j + 2], struct.unpack(">H", block[j + 3:j + 5])[0]
        if rec == 2 and ds in IPTC_TAGS:
            _add(fields, f"iptc:{IPTC_TAGS[ds]}", block[j + 5:j + 5 + n].decode("utf-8", "replace"))
        j += 5 + n


def _png_fields(path: Path, fields: dict) -> tuple[int, int] | None:
    size = None
    with open(path, "rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            return None
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            n, kind = struct.unpack(">I4s", head)
            if kind == b"IEND" or (kind == b"IDAT" and "png:prompt" in fields):
                break       # ComfyUI writes its text before the image data: no need to walk every IDAT chunk
            if kind == b"caBX":
                _add(fields, "c2pa", "")
            if kind not in (b"IHDR", b"tEXt", b"zTXt", b"iTXt", b"eXIf"):
                f.seek(n + 4, 1)
                continue
            data = f.read(n)
            f.seek(4, 1)
            try:
                if kind == b"IHDR":
                    size = struct.unpack(">II", data[:8])
                elif kind == b"eXIf":
                    _exif(data, fields)
                elif kind == b"tEXt":
                    k, v = data.split(b"\0", 1)
                    _add(fields, "png:" + k.decode("latin-1"), v.decode("utf-8", "replace"))
                elif kind == b"zTXt":
                    k, v = data.split(b"\0", 1)
                    _add(fields, "png:" + k.decode("latin-1"), zlib.decompress(v[1:]).decode("utf-8", "replace"))
                else:
                    k, v = data.split(b"\0", 1)
                    compressed, v = v[0], v[2:]
                    _lang, v = v.split(b"\0", 1)
                    _tkey, v = v.split(b"\0", 1)
                    v = zlib.decompress(v) if compressed else v
                    if k == b"XML:com.adobe.xmp":
                        _xmp(v, fields)
                    else:
                        _add(fields, "png:" + k.decode("latin-1"), v.decode("utf-8", "replace"))
            except (ValueError, IndexError, zlib.error, struct.error):
                continue
    return size


def _jpeg_fields(f, fields: dict) -> None:
    f.seek(2)
    while True:
        h = f.read(4)
        if len(h) < 4 or h[0] != 0xFF or h[1] in (0xDA, 0xD9):
            break
        (n,) = struct.unpack(">H", h[2:])
        data = f.read(n - 2)
        if h[1] == 0xE1 and data.startswith(b"Exif\0"):
            _exif(data, fields)
        elif h[1] == 0xE1 and data.startswith(b"http://ns.adobe.com/xap/1.0/\0"):
            _xmp(data, fields)
        elif h[1] == 0xED and data.startswith(b"Photoshop 3.0\0"):
            _iptc(data, fields)
        elif h[1] == 0xEB and b"jumb" in data[:32]:
            _add(fields, "c2pa", "")
        elif h[1] == 0xFE:
            _add(fields, "jpeg:comment", data.decode("utf-8", "replace"))


def _webp_fields(f, fields: dict) -> None:
    f.seek(12)
    while True:
        h = f.read(8)
        if len(h) < 8:
            break
        kind, n = h[:4], struct.unpack("<I", h[4:])[0]
        if kind in (b"EXIF", b"XMP ", b"C2PA"):
            data = f.read(n)
            f.seek(n & 1, 1)
            if kind == b"EXIF":
                _exif(data, fields)
            elif kind == b"XMP ":
                _xmp(data, fields)
            else:
                _add(fields, "c2pa", "")
        else:
            f.seek(n + (n & 1), 1)


def _video_fields(path: Path, fields: dict) -> tuple[int, int] | None:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe is not installed, so video files cannot be read.")
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        "-select_streams", "v:0", str(path)], capture_output=True, timeout=60)
    if r.returncode:
        raise RuntimeError(r.stderr.decode("utf-8", "replace").strip() or "ffprobe could not read the file.")
    info = json.loads(r.stdout or b"{}")
    for k, v in (info.get("format", {}).get("tags") or {}).items():
        _add(fields, "video:" + k.lower(), str(v))
    stream = (info.get("streams") or [{}])[0]
    return (stream["width"], stream["height"]) if stream.get("width") else None


def read_fields(path: Path) -> tuple[dict, tuple[int, int] | None]:
    """Every text metadata field of one output, and its (width, height) when known."""
    fields, ext = {}, path.suffix.lower()
    if ext == ".png":
        return fields, _png_fields(path, fields)
    if ext in VIDEO_EXT:
        return fields, _video_fields(path, fields)
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            _webp_fields(f, fields)
        elif head[:2] == b"\xff\xd8":
            _jpeg_fields(f, fields)
    return fields, None


def _json_after(text: str, label: str):
    """The JSON value written after 'label:' (ComfyUI's EXIF form, e.g. 'prompt:{...}'), or None."""
    for m in re.finditer(re.escape(label) + r"\s*:\s*", text, re.I):
        try:
            return json.JSONDecoder().raw_decode(text, m.end())[0]
        except ValueError:
            continue
    return None


def _as_graph(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def comfy_graphs(fields: dict) -> tuple[dict | None, dict | None]:
    """(prompt graph, workflow) when the fields hold ComfyUI's data, else (None, None)."""
    prompt = _as_graph(fields.get("png:prompt")) or _as_graph(fields.get("video:prompt"))
    workflow = _as_graph(fields.get("png:workflow")) or _as_graph(fields.get("video:workflow"))
    comment = _as_graph(fields.get("video:comment"))       # VideoHelperSuite writes {"prompt": ..., "workflow": ...}
    if comment:
        prompt = prompt or _as_graph(comment.get("prompt"))
        workflow = workflow or _as_graph(comment.get("workflow"))
    if not prompt:                                          # ComfyUI's WebP/JPEG form: EXIF text "prompt:{...}"
        exif = "\n".join(v for k, v in fields.items() if k.startswith("exif:"))
        prompt, workflow = _as_graph(_json_after(exif, "prompt")), workflow or _as_graph(_json_after(exif, "workflow"))
    if prompt and not all(isinstance(v, dict) and "class_type" in v for v in prompt.values()):
        return None, None                                   # a "prompt" field that is not a ComfyUI graph
    return prompt, workflow


def read_metadata(path: Path) -> tuple[dict | None, dict | None, tuple[int, int] | None]:
    """(ComfyUI prompt graph, workflow, (width, height)) of one output; graphs are None when it carries none."""
    fields, size = read_fields(path)
    return (*comfy_graphs(fields), size)


# ---------------------------------------------------------------- prompts outside ComfyUI
#
# Other generators and providers sometimes leave the prompt in a description or comment field. A field counts only
# when its name says it can hold a prompt and its text reads like one: titles, camera, software, author and copyright
# fields never do, and short labels, links, IDs and credit lines are skipped. XMP fields of namespaces not listed in
# XMP_PREFIX count only when their name says "prompt". A file whose metadata marks it as a photograph or a
# rights-managed scan (rights, author or camera fields) gives no prompts unless it also carries an AI marker (C2PA,
# an AIGC label, or an IPTC digital source type of algorithmic media): museum and stock captions read like prompts.
# A JSON field is read for its prompt, negative prompt, seed and model keys. "Prompt / Negative prompt: / Steps: …"
# text is split into its parts.

PROMPT_FIELD = re.compile(r"description|comment|caption|prompt|parameters|subject|instructions|abstract|synopsis|"
                          r"summary|dream|input", re.I)
PHOTO_SIGN = re.compile(r"^(?:xmp:xmpRights:|xmp:dc:(?:rights|creator)$|iptc:(?:Byline|Copyright)$|"
                        r"exif:(?:Copyright|Artist|Make|Model|CameraOwnerName)$|xmp:tiff:(?:Make|Model)$|xmp:aux:)")
KNOWN_XMP = {f"xmp:{p}:" for p in XMP_PREFIX.values()}
NOT_PROMPT_FIELD = re.compile(r"make|model|software|artist|author|byline|copyright|rights|creatortool|date|time|"
                              r"encoder|handler|brand|documentid|instanceid|xmpmm|history|writer|credit|source|"
                              r"keywords|owner|c2pa|workflow|exif:0x|dc:subject", re.I)
JUNK_TEXT = re.compile(r"^(?:\W*(?:©|\(c\)|copyright\b|all rights reserved|https?://\S+$|www\.\S+$|"
                       r"(?:made|created|generated|edited|produced) (?:with|by|using)\b|ai[- ]generated\b|"
                       r"screenshot\b|untitled\b|image\b \d|img_\d|dsc_?\d|lavf\d|isom\b|[0-9a-f-]{32,}$))", re.I)
JSON_PROMPT = ("prompt", "positive_prompt", "text_prompt", "input_prompt", "caption", "description")
JSON_NEGATIVE = ("negative_prompt", "uc", "negative")
JSON_MODEL = ("model", "model_name", "model_id", "base_model", "sd_model_name")


def _looks_like_prompt(text: str) -> bool:
    return len(re.findall(r"[^\W\d_]{2,}", text)) >= 4 and not JUNK_TEXT.match(text.strip())


def _settings_text(text: str) -> dict | None:
    """'prompt\\nNegative prompt: …\\nSteps: 30, Sampler: …, Seed: 123, Model: …' split into its parts."""
    m = re.search(r"\n(?:Negative prompt:|Steps: \d)", text)
    if not m:
        return None
    prompt, rest = text[:m.start()].strip(), text[m.start():].strip()
    neg = re.match(r"Negative prompt:\s*(.*?)(?:\n(?=Steps: \d)|$)", rest, re.S)
    settings = rest[neg.end():] if neg else rest
    pairs = dict(re.findall(r"(\w[\w ]*?): ([^,\n]+)", settings))
    out = {"text": prompt, "negative": neg[1].strip() if neg else ""}
    if pairs.get("Seed", "").isdigit():
        out["seed"] = int(pairs["Seed"])
    if pairs.get("Model"):
        out["models"] = [pairs["Model"].strip()]
    for key, name in (("Steps", "steps"), ("Sampler", "sampler"), ("CFG scale", "cfg")):
        if key in pairs:
            v = pairs[key].strip()
            out[name] = float(v) if re.fullmatch(r"\d+\.\d+", v) else int(v) if v.isdigit() else v
    return out


def _ai_marked(fields: dict) -> bool:
    return any(name == "c2pa" or "aigc" in name.lower() or
               (name.endswith("DigitalSourceType") and "algorithmic" in value.lower()) for name, value in fields.items())


def other_prompts(fields: dict) -> list[dict]:
    """Prompts found in a file's non-ComfyUI metadata: [{text, source, negative?, seed?, models?, …}]."""
    found, seen = [], set()
    if any(PHOTO_SIGN.match(name) for name in fields) and not _ai_marked(fields):
        return found
    for name, value in fields.items():
        if not value or NOT_PROMPT_FIELD.search(name) or (
                name.startswith("xmp:") and name[:name.index(":", 4) + 1] not in KNOWN_XMP
                and "prompt" not in name.lower()):
            continue
        data = _as_graph(value) if value.lstrip().startswith("{") else None
        if data is not None:
            text = next((data[k] for k in JSON_PROMPT if isinstance(data.get(k), str) and data[k].strip()), None)
            if not text:
                continue
            item = {"text": text.strip()}
            neg = next((data[k] for k in JSON_NEGATIVE if isinstance(data.get(k), str) and data[k].strip()), None)
            if neg:
                item["negative"] = neg.strip()
            if isinstance(data.get("seed"), int) and not isinstance(data["seed"], bool):
                item["seed"] = data["seed"]
            model = next((data[k] for k in JSON_MODEL if isinstance(data.get(k), str) and data[k].strip()), None)
            if model:
                item["models"] = [model.strip()]
        elif PROMPT_FIELD.search(name):
            item = _settings_text(value) or {"text": re.sub(r"\s*Job ID:\s*[0-9a-f-]{36}\s*$", "", value).strip()}
        else:
            continue
        if not _looks_like_prompt(item["text"]) or norm(item["text"]) in seen:
            continue
        seen.add(norm(item["text"]))
        found.append({**item, "source": name})
    return found


# ---------------------------------------------------------------- understanding the graph

def norm(text: str) -> str:
    """A prompt's identity: runs of white space as one space, ends trimmed (as Mondocraft compares prompts)."""
    return " ".join(str(text or "").split())


def _link(value) -> str | None:
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int):
        return str(value[0])
    return None


def _upstream(graph: dict, start: str | None, skip: tuple = ()) -> list[str]:
    """Node ids reachable upstream from start (start included), not following inputs named in skip."""
    if start is None or start not in graph:
        return []
    seen, todo = [], [start]
    while todo:
        nid = todo.pop()
        if nid in seen or nid not in graph:
            continue
        seen.append(nid)
        for key, value in (graph[nid].get("inputs") or {}).items():
            src = _link(value)
            if src and key not in skip:
                todo.append(src)
    return seen


def _strings(graph: dict, ids: list[str]) -> list[str]:
    out = []
    for nid in ids:
        node = graph[nid]
        if NOTE_TYPES.search(node.get("class_type", "")):
            continue
        inputs = node.get("inputs") or {}
        keys = ("populated_text",) if isinstance(inputs.get("populated_text"), str) and inputs["populated_text"].strip() \
            else TEXT_KEYS
        for key in keys:
            value = inputs.get(key)
            if isinstance(value, str) and value.strip() and not value.lower().endswith(MODEL_EXT):
                out.append(value.strip())
    return list(dict.fromkeys(out))


def _is_sampler(node: dict) -> bool:
    inputs = node.get("inputs") or {}
    return "latent_image" in inputs and any(k in inputs for k in ("positive", "guider", "conditioning"))


def _output_nodes(graph: dict, filename: str) -> list[str]:
    """The save nodes that wrote this file: the longest filename_prefix the name starts with; all save nodes when
    none matches."""
    saves = {nid: n for nid, n in graph.items()
             if "filename_prefix" in (n.get("inputs") or {}) and not n.get("class_type", "").startswith("Preview")}
    stem = Path(filename).stem
    best = [(len(p), nid) for nid, n in saves.items()
            if isinstance(p := n["inputs"]["filename_prefix"], str) and (p := p.replace("\\", "/").split("/")[-1])
            and stem.startswith(p)]
    if best:
        top = max(best)[0]
        return [nid for length, nid in best if length == top]
    return list(saves) or list(graph)


def _sampler_parts(graph: dict, nid: str) -> dict:
    """The positive/negative source, model source, seed and settings of one sampler node."""
    node = graph[nid]
    inputs = node.get("inputs") or {}
    guider = graph.get(_link(inputs.get("guider")) or "", {}).get("inputs") or {}
    pos = _link(inputs.get("positive")) or _link(guider.get("positive")) or _link(guider.get("conditioning")) \
        or _link(inputs.get("conditioning"))
    neg = _link(inputs.get("negative")) or _link(guider.get("negative"))
    model = _link(inputs.get("model")) or _link(guider.get("model"))
    seed = None
    for holder in (inputs, graph.get(_link(inputs.get("noise")) or "", {}).get("inputs") or {}):
        for key in ("seed", "noise_seed"):
            v = holder.get(key)
            if isinstance(v, int):
                seed = v
            elif _link(v):             # a seed node feeding the sampler
                src = graph.get(_link(v), {}).get("inputs") or {}
                seed = next((x for k, x in src.items() if isinstance(x, int) and not isinstance(x, bool)
                             and ("seed" in k.lower() or k == "value")), seed)
        if seed is not None:
            break
    sched = {}
    sigmas = graph.get(_link(inputs.get("sigmas")) or "", {}).get("inputs") or {}
    sampler = graph.get(_link(inputs.get("sampler")) or "", {}).get("inputs") or {}
    for key, src in (("sampler", inputs.get("sampler_name", sampler.get("sampler_name"))),
                     ("scheduler", inputs.get("scheduler", sigmas.get("scheduler"))),
                     ("steps", inputs.get("steps", sigmas.get("steps"))),
                     ("cfg", inputs.get("cfg", guider.get("cfg"))),
                     ("denoise", inputs.get("denoise", sigmas.get("denoise")))):
        if isinstance(src, (int, float, str)) and not isinstance(src, bool):
            sched[key] = src
    for nid in _upstream(graph, pos, ("clip", "model", "vae")):   # Flux-style guidance on the positive conditioning
        g = (graph[nid].get("inputs") or {}).get("guidance")
        if isinstance(g, (int, float)) and not isinstance(g, bool):
            sched["guidance"] = g
            break
    return dict(pos=pos, neg=neg, model=model, seed=seed, settings=sched)


def _loras(graph: dict, ids: list[str]) -> list[dict]:
    """LoRAs switched on in the given nodes, in graph order: {name, strength[, clip_strength]}."""
    out = {}
    for nid in reversed(ids):          # upstream walk lists the sampler end first; loaders nearest the model last
        node = graph[nid]
        inputs = node.get("inputs") or {}
        name = inputs.get("lora_name")
        if isinstance(name, str) and name not in ("", "None"):
            strength = inputs.get("strength_model", inputs.get("strength", 1.0))
            clip = inputs.get("strength_clip")
            if strength or clip:
                out[name] = dict(name=name, strength=strength,
                                 **({"clip_strength": clip} if isinstance(clip, (int, float)) and clip != strength else {}))
        for key, value in inputs.items():
            if isinstance(value, dict) and isinstance(value.get("lora"), str):   # rgthree Power Lora Loader
                if value.get("on") and value["lora"] not in ("", "None") and value.get("strength", 1):
                    two = value.get("strengthTwo")
                    out[value["lora"]] = dict(name=value["lora"], strength=value.get("strength", 1),
                                              **({"clip_strength": two} if isinstance(two, (int, float)) else {}))
            m = re.fullmatch(r"lora_(\d+)", key)                                  # rgthree Lora Loader Stack and similar
            if m and isinstance(value, str) and value not in ("", "None"):
                strength = next((inputs[k] for k in (f"strength_{m[1]}", f"lora_wt_{m[1]}", f"model_str_{m[1]}")
                                 if k in inputs), 1.0)
                if strength:
                    out[value] = dict(name=value, strength=strength)
    return list(out.values())


# A LoRA's trigger words are often in its file name ("ohwx_woman_v2.safetensors" is triggered by "ohwx woman"), and
# sit at the start of the prompt. Without the LoRA they only pollute the prompt, so they are taken off and kept with
# the LoRA instead. Name parts that say only what the file is (version, base model, kind) never count as a trigger.
LORA_NAME_NOISE = re.compile(r"(?:v|\d+|v\d+|e\d+|r\d+|rank\d+|epoch\d*|step\d*|fp\d+|bf\d+|lora|lycoris|locon|loha|lokr|"
                             r"dora|flux\d*|f1|dev|schnell|sd\d*|sdxl|xl|pony|illustrious|il|noob|sd3|wan\d*|hunyuan|hv|"
                             r"ltx|qwen|chroma|t2v|i2v|ti2v|trigger|style|by)")
PROMPT_SEGMENT = re.compile(r"\s*(.+?)\s*(?:[,;\n]|\.(?=\s)|$)", re.S)


def _name_tokens(name: str) -> list[str]:
    """A LoRA file name's words: "flux/OhwxWoman_v2.safetensors" -> ["ohwx", "woman", "v", "2"]."""
    base = re.split(r"[\\/]", name)[-1]
    if base.lower().endswith(MODEL_EXT):
        base = base.rsplit(".", 1)[0]
    base = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", base)
    return re.findall(r"[a-z0-9]+", base.lower())


def _is_trigger(segment: str, tokens: list[str]) -> bool:
    """Whether a short piece of prompt is spelled by consecutive words of a LoRA's name, separators aside:
    "ohwx woman", "Pixel Art" and "pixelart" all match "pixel_art_xl"/"ohwx_woman"; "art" does not match "smart"."""
    words = re.findall(r"[a-z0-9]+", re.sub(r":\s*-?\d+(?:\.\d+)?\s*\)", ")", segment.lower()))  # (word:1.2)
    want = "".join(words)
    if not 1 <= len(words) <= 4 or len(want) < 3:
        return False
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, len(tokens)):
            joined += tokens[j]
            if joined == want:
                return not all(LORA_NAME_NOISE.fullmatch(t) for t in tokens[i:j + 1])
            if not want.startswith(joined):
                break
    return False


def strip_triggers(text: str, loras: list[dict]) -> tuple[str, dict]:
    """The prompt without the LoRA trigger words at its start, and {LoRA name: trigger} for those taken off.
    Only whole comma-separated pieces are taken, and only while what is left still reads as a prompt."""
    names = [(lora["name"], _name_tokens(lora["name"])) for lora in loras]
    rest, found = text, {}
    while names:
        m = PROMPT_SEGMENT.match(rest)
        remaining = rest[m.end():]
        hit = next((name for name, tokens in names if _is_trigger(m[1], tokens)), None) if remaining.strip() else None
        if hit is None:
            break
        found[hit] = f"{found[hit]}, {m[1]}" if hit in found else m[1]
        rest = remaining
    rest = rest.lstrip(" \t\n,;.")
    if not found or not _prompt_like(rest):
        return text, {}
    return rest, found


def _names(graph: dict, ids: list[str], keys: tuple) -> list[str]:
    out = []
    for nid in reversed(ids):
        inputs = graph[nid].get("inputs") or {}
        out += [inputs[k] for k in keys if isinstance(inputs.get(k), str) and inputs[k] not in ("", "None")]
    return list(dict.fromkeys(out))


def _workflow_strings(workflow: dict | None) -> list[str]:
    """Prompt-like text in the saved workflow's text, prompt and display nodes, including muted or bypassed ones."""
    out = []
    if not isinstance(workflow, dict):
        return out
    nodes = list(workflow.get("nodes") or [])
    for sub in (workflow.get("definitions") or {}).get("subgraphs") or []:
        nodes += sub.get("nodes") or []
    for node in nodes:
        kind = str(node.get("type", ""))
        if NOTE_TYPES.search(kind) or not TEXT_NODE_TYPES.search(kind):
            continue
        stack = [node.get("widgets_values")]
        while stack:
            v = stack.pop()
            if isinstance(v, list):
                stack += v
            elif isinstance(v, dict):
                stack += list(v.values())
            elif isinstance(v, str) and v.strip():
                out.append(v.strip())
    return list(dict.fromkeys(out))


def _prompt_like(text: str) -> bool:
    return len(re.findall(r"\w+", text)) >= 3 and not text.lower().endswith(MODEL_EXT) and "://" not in text[:12]


def analyse(graph: dict, workflow: dict | None, filename: str) -> dict:
    """What one output's ComfyUI data says: used prompts, other prompt-like text, negative, models, LoRAs, seeds."""
    outputs = _output_nodes(graph, filename)
    live = list(dict.fromkeys(nid for out in outputs for nid in _upstream(graph, out)))
    samplers = [nid for nid in live if _is_sampler(graph[nid])]
    # The first pass is the sampler with the fewest samplers upstream of it.
    samplers.sort(key=lambda s: (sum(_is_sampler(graph[u]) for u in _upstream(graph, s)), int(s) if s.isdigit() else 0))
    parts = [_sampler_parts(graph, s) for s in samplers]
    skip = ("clip", "model", "vae")
    positives = list(dict.fromkeys(t for p in parts for t in _strings(graph, _upstream(graph, p["pos"], skip))))
    negative = "\n".join(_strings(graph, _upstream(graph, parts[0]["neg"], skip))) if parts else ""
    negatives = {norm(t) for p in parts for t in _strings(graph, _upstream(graph, p["neg"], skip))}
    model_ids = _upstream(graph, parts[0]["model"]) if parts else live
    models = _names(graph, model_ids, CHECKPOINT_KEYS) + [m for m in _names(graph, live, CHECKPOINT_KEYS)
                                                          if m not in _names(graph, model_ids, CHECKPOINT_KEYS)]
    clip_ids = [nid for p in parts for nid in _upstream(graph, p["pos"])]
    used = {norm(t) for t in positives} | negatives
    others = [t for t in _strings(graph, list(graph)) + _workflow_strings(workflow)
              if _prompt_like(t) and norm(t) not in used]
    seeds = [p["seed"] for p in parts if p["seed"] is not None]
    video_out = any(VIDEO_OUTPUT.search(graph[o].get("class_type", "")) for o in outputs if o in graph)
    return dict(
        positives=positives, others=list(dict.fromkeys(others)), negative=negative, negatives=negatives,
        models=models, text_encoders=_names(graph, clip_ids, ENCODER_KEYS) or _names(graph, live, ENCODER_KEYS),
        loras=_loras(graph, model_ids + [n for n in clip_ids if n not in model_ids]),
        settings=parts[0]["settings"] if parts else {}, passes=len(parts), seeds=seeds,
        t2i=not any(LOAD_MEDIA.match(graph[n].get("class_type", "")) for n in live),
        video=video_out)


# ---------------------------------------------------------------- crawling

def _date(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def _setup_key(setup: dict) -> str:
    return json.dumps({k: v for k, v in setup.items() if k not in ("runs", "seeds")}, sort_keys=True)


def _is_audio_copy(path: Path) -> bool:
    """Video Combine's copy of a video with its sound added: clip_00001-audio.mp4 beside clip_00001.mp4."""
    return path.suffix.lower() in VIDEO_EXT and path.stem.endswith("-audio")


def _output_stem(path: Path, folder: Path) -> str:
    """The name shared by every file of one output: a video, its first-frame PNG and its -audio copy."""
    stem = path.with_suffix("").relative_to(folder)
    return str(stem.with_name(stem.name[:-len("-audio")]) if _is_audio_copy(path) else stem)


def _file_id(path: Path) -> list:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def _load_partial(checkpoint: Path | None, folder: Path, recurse: bool, files: list[Path], log) -> dict | None:
    """The saved progress of an earlier, unfinished crawl of this folder, or None to start fresh. Progress is only
    used when every file it read is still there unchanged; files that failed to read are read again."""
    if checkpoint is None or not checkpoint.exists():
        return None
    try:
        state = json.loads(checkpoint.read_text())
    except (OSError, ValueError):
        log("  The saved progress could not be read; starting from the beginning.")
        return None
    if state.get("format") != PARTIAL_FORMAT or state.get("folder") != str(folder) or state.get("recursive") != recurse:
        return None
    present = {str(p.relative_to(folder)): p for p in files}
    try:
        if any(rel not in present or _file_id(present[rel]) != fid for rel, fid in state["seen"].items()):
            log("  Files read before have changed since; starting from the beginning.")
            return None
    except OSError:
        return None
    for rel in state["seen"]:             # progress saved before 1.2.0 kept an -audio copy under its own name
        stem = str(Path(rel).with_suffix(""))
        if _is_audio_copy(Path(rel)) and stem in state.get("done_stems", []):
            state["done_stems"].append(_output_stem(folder / rel, folder))
    log(f"  Resuming: {len(state['seen'])} files were read in an earlier run.")
    return state


def _save_partial(checkpoint: Path, state: dict) -> None:
    tmp = checkpoint.with_name(checkpoint.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(checkpoint)


def crawl(folder: Path, recurse: bool = True, log=print, checkpoint: Path | None = None,
          save_every: float = 120.0, keep_triggers: bool = False) -> dict:
    """Harvest one folder (and its subfolders unless recurse is False) into a harvest dict.
    With a checkpoint file, progress is saved there every save_every seconds (and on Ctrl+C), and a later crawl of
    the same folder carries on from it. The caller removes the checkpoint once the harvest is written.
    LoRA trigger words are taken off the start of ComfyUI prompts unless keep_triggers is True."""
    folder = folder.resolve()
    files = sorted((p for p in (folder.rglob("*") if recurse else folder.iterdir())
                    if p.is_file() and p.suffix.lower() in IMAGE_EXT + VIDEO_EXT),
                   key=lambda p: (str(p.parent), p.suffix.lower() not in VIDEO_EXT, _is_audio_copy(p), p.name))
    state = _load_partial(checkpoint, folder, recurse, files, log) or {}
    if state and state.get("keep_triggers", True) != keep_triggers:   # progress from 1.0.0 kept every trigger
        state = {}
    c = Crawl(records={(k, n): rec for k, n, rec in state.get("records", [])}, seen=state.get("seen", {}),
              unread=state.get("unread", []), done_stems=set(state.get("done_stems", [])),
              negatives=set(state.get("negatives", [])), fields=state.get("fields", {}), keep_triggers=keep_triggers)
    saved_at = time.time()

    def save():
        _save_partial(checkpoint, dict(format=PARTIAL_FORMAT, folder=str(folder), recursive=recurse, seen=c.seen,
                                       records=[[k, n, rec] for (k, n), rec in c.records.items()], unread=c.unread,
                                       done_stems=sorted(c.done_stems), negatives=sorted(c.negatives),
                                       fields=c.fields, keep_triggers=keep_triggers))

    try:
        for i, path in enumerate(files, 1):
            if i % 500 == 0:
                log(f"  {i}/{len(files)} files read")
            if checkpoint is not None and time.time() - saved_at >= save_every:
                save()
                saved_at = time.time()
            c.harvest(folder, path)
    except KeyboardInterrupt:
        if checkpoint is not None:
            save()
            log(f"\n  Stopped. Progress is saved in {checkpoint}; run the same crawl again to carry on.")
        raise
    prompts = []
    for (kind, key), rec in c.records.items():
        if not rec["setups"] and key in c.negatives:
            continue                    # an unused text that is some run's negative prompt is not a prompt
        prompts.append(dict(text=rec["text"], kind=kind,
                            **({"t2i": rec["t2i"]} if kind == "image" and rec["t2i"] is not None else {}),
                            first_seen=_date(rec["first"]), last_seen=_date(rec["last"]),
                            first_ts=round(rec["first"]), last_ts=round(rec["last"]),
                            images=sum(len(s["runs"]) for s in rec["setups"]), setups=rec["setups"],
                            **({"unused_in": rec["unused_in"]} if rec["unused_in"] else {})))
    prompts.sort(key=lambda p: (p["first_ts"], p["text"]))
    triggers = {}
    for p in prompts:
        for lora, trigger in {(lora["name"], lora["trigger"]) for s in p["setups"] for lora in s.get("loras", [])
                              if "trigger" in lora}:
            triggers[lora, trigger] = triggers.get((lora, trigger), 0) + 1
    fields = {k: dict(files=v[0], prompts=v[1]) for k, v in sorted(c.fields.items(), key=lambda kv: (-kv[1][0], kv[0]))}
    return dict(format=FORMAT, folder=str(folder), recursive=recurse,
                harvested_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                files_scanned=len(files), files_with_prompts=len(c.done_stems), prompts_found=len(prompts),
                metadata_fields=fields, lora_triggers=[dict(lora=lora, trigger=t, prompts=n) for (lora, t), n in
                                                      sorted(triggers.items(), key=lambda kv: (-kv[1], kv[0]))],
                files_without_prompts=c.unread, errors=c.errors, prompts=prompts)


class Crawl:
    """The running state of one crawl: prompt records keyed by (kind, normalised text), and bookkeeping."""

    def __init__(self, records, seen, unread, done_stems, negatives, fields, keep_triggers=False):
        self.records, self.seen, self.unread, self.errors = records, seen, unread, []
        self.keep_triggers = keep_triggers
        self.done_stems, self.negatives, self.fields = done_stems, negatives, fields   # fields: name -> [files, prompts]

    def harvest(self, folder: Path, path: Path) -> None:
        """Read one output and add what it holds."""
        rel = str(path.relative_to(folder))
        stem = _output_stem(path, folder)
        if rel in self.seen or stem in self.done_stems:  # read before, or a video's first-frame PNG or -audio copy
            return
        try:
            fields, size = read_fields(path)
            st = path.stat()
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as e:
            self.errors.append({"file": rel, "error": str(e)})
            return
        self.seen[rel] = [st.st_size, st.st_mtime_ns]
        for name in fields:
            self.fields.setdefault(name, [0, 0])[0] += 1
        run = {"file": rel, "date": _date(st.st_mtime)}
        size = dict(width=size[0], height=size[1]) if size else {}
        graph, workflow = comfy_graphs(fields)
        try:
            found = self._comfy(graph, workflow, fields, path, run, size, st.st_mtime) if graph else \
                self._other(fields, path, run, size, st.st_mtime)
        except Exception as e:  # one odd file must not stop a long crawl
            self.errors.append({"file": rel, "error": f"{type(e).__name__}: {e}"})
            return
        if not found:
            self.unread.append(rel)
            return
        self.done_stems.add(stem)

    def _record(self, kind, text, t2i, mtime):
        rec = self.records.setdefault((kind, norm(text)), dict(text=text, kind=kind, t2i=t2i, setups=[],
                                                               unused_in=[], first=mtime, last=mtime))
        rec["t2i"] = True if True in (rec["t2i"], t2i) else rec["t2i"] if rec["t2i"] is not None else t2i
        rec["first"], rec["last"] = min(rec["first"], mtime), max(rec["last"], mtime)
        return rec

    def _use(self, kind, text, t2i, setup, run, mtime):
        rec = self._record(kind, text, t2i, mtime)
        key = _setup_key(setup)
        group = next((s for s in rec["setups"] if _setup_key(s) == key), None)
        if group is None:
            rec["setups"].append(group := {**setup, "runs": []})
        group["runs"].append(run)

    def _comfy(self, graph, workflow, fields, path, run, size, mtime) -> bool:
        info = analyse(graph, workflow, path.name)
        if not info["positives"] and not info["others"]:
            return False
        holder = next((n for n in ("png:prompt", "video:prompt", "video:comment") if n in fields), None) or \
            next((n for n, v in fields.items() if n.startswith("exif:") and "prompt:" in v[:40].lower()), None)
        if holder:
            self.fields[holder][1] += 1
        kind = "video" if path.suffix.lower() in VIDEO_EXT or info["video"] else "image"
        setup = {k: v for k, v in dict(source="comfyui", models=info["models"], text_encoders=info["text_encoders"],
                                       loras=info["loras"], negative=info["negative"], **info["settings"],
                                       passes=info["passes"] if info["passes"] > 1 else None, **size).items()
                 if v not in (None, [], "")}
        if info["seeds"]:
            run["seed"] = info["seeds"][0]
            if len(info["seeds"]) > 1:
                run["more_seeds"] = info["seeds"][1:]
        used = set()
        for text in info["positives"]:
            text, triggers = (text, {}) if self.keep_triggers else strip_triggers(text, info["loras"])
            used.add(norm(text))
            self._use(kind, text, info["t2i"], {**setup, "loras": [
                {**lora, "trigger": triggers[lora["name"]]} if lora["name"] in triggers else lora
                for lora in setup["loras"]]} if triggers else setup, run, mtime)
        for text in info["others"]:
            text = text if self.keep_triggers else strip_triggers(text, info["loras"])[0]
            if norm(text) not in used:
                self._record(kind, text, True, mtime)["unused_in"].append(run["file"])
        self.negatives.update(info["negatives"])
        return True

    def _other(self, fields, path, run, size, mtime) -> bool:
        found = other_prompts(fields)
        kind = "video" if path.suffix.lower() in VIDEO_EXT else "image"
        for item in found:
            self.fields[item["source"]][1] += 1
            setup = {k: v for k, v in dict(source=item["source"], models=item.get("models"),
                                           negative=item.get("negative"), steps=item.get("steps"),
                                           sampler=item.get("sampler"), cfg=item.get("cfg"), **size).items()
                     if v not in (None, [], "")}
            this_run = {**run, **({"seed": item["seed"]} if "seed" in item else {})}
            self._use(kind, item["text"], None, setup, this_run, mtime)
            if item.get("negative"):
                self.negatives.add(norm(item["negative"]))
        return bool(found)


def harvest_path(folder: Path, out_dir: Path) -> Path:
    """harvests/<folder name>.json; a different folder with the same name gets its path's short hash added."""
    folder = folder.resolve()
    name = re.sub(r"[^\w.-]+", "_", folder.name or "root").strip("._") or "root"
    path = out_dir / f"{name}.json"
    for taken in (path, partial_path(path)):      # a finished or unfinished harvest of another folder of that name
        try:
            if taken.exists() and json.loads(taken.read_text()).get("folder") != str(folder):
                return out_dir / f"{name}-{hashlib.sha1(str(folder).encode()).hexdigest()[:6]}.json"
        except (OSError, ValueError):
            pass
    return path


def partial_path(harvest: Path) -> Path:
    """Where an unfinished crawl keeps its progress: <harvest file>.partial."""
    return harvest.with_name(harvest.name + ".partial")


# ---------------------------------------------------------------- merging into a prompt catalog

def _catalog_setup(setup: dict) -> dict:
    """A harvested setup as kept in the catalog: its runs reduced to their seeds."""
    out = {"source": "comfyui", **{k: v for k, v in setup.items() if k != "runs"}}
    seeds = [r["seed"] for r in setup.get("runs", []) if "seed" in r]
    if seeds:
        out["seeds"] = sorted(set(seeds))
    return out


def _upgrade(items: list[dict]) -> int:
    """Catalog entries merged by the first version keep their data under "comfyui": move it to "harvest"."""
    moved = 0
    for p in items:
        if isinstance(p.get("comfyui"), dict) and "harvest" not in p:
            p["harvest"] = p.pop("comfyui")
            for s in p["harvest"].get("setups", []):
                s.setdefault("source", "comfyui")
            moved += 1
    return moved


def merge_harvest(items: list[dict], harvest: dict) -> dict:
    """Merge one harvest into a Mondocraft catalog list in place. Returns {"added": n, "updated": n, "unchanged": n}.
    New prompts get an id, the harvest's first and last file dates as created and last_used, and their data under
    "harvest"; prompts already in the catalog (same kind, same text apart from white space) gain the data instead."""
    if harvest.get("format") not in (FORMAT, *OLD_FORMATS):
        raise ValueError("Not a mondo-harvest file.")
    _upgrade(items)
    index = {(p.get("kind") or "image", norm(p.get("text", ""))): p for p in items}
    counts = dict(added=0, updated=0, unchanged=0)
    for h in harvest.get("prompts", []):
        text, kind = h["text"].strip(), h.get("kind", "image")
        if not text:
            continue
        data = {"folders": [harvest["folder"]], "setups": [_catalog_setup(s) for s in h.get("setups", [])]}
        if h.get("unused_in"):
            data["unused_in_workflows"] = len(h["unused_in"])
        t2i = h.get("t2i") if kind == "image" and isinstance(h.get("t2i"), bool) else None
        p = index.get((kind, norm(text)))
        if p is None:
            p = dict(id=uuid.uuid4().hex[:8], text=text, model="", kind=kind, created=h["first_ts"],
                     last_used=h["last_ts"], **({"t2i": t2i} if t2i is not None else {}), harvest=data)
            items.append(p)
            index[(kind, norm(text))] = p
            counts["added"] += 1
            continue
        before = json.dumps(p, sort_keys=True)
        old = p.setdefault("harvest", {"folders": [], "setups": []})
        if "unused_in_workflows" in data and harvest["folder"] not in old.get("folders", []):
            old["unused_in_workflows"] = old.get("unused_in_workflows", 0) + data["unused_in_workflows"]
        old["folders"] = list(dict.fromkeys(old.get("folders", []) + data["folders"]))
        setups = old.setdefault("setups", [])
        for s in data["setups"]:
            key = _setup_key(s)
            same = next((x for x in setups if _setup_key(x) == key), None)
            if same is None:
                setups.append(s)
            elif s.get("seeds"):
                same["seeds"] = sorted(set(same.get("seeds", [])) | set(s["seeds"]))
        p["created"] = min(p.get("created", h["first_ts"]), h["first_ts"])
        p["last_used"] = max(p.get("last_used", 0), h["last_ts"])
        if t2i:
            p["t2i"] = True
        counts["updated" if json.dumps(p, sort_keys=True) != before else "unchanged"] += 1
    return counts


def load_mondocraft(folder: Path | None = None):
    """Mondocraft's server module, imported from its folder: its own prompt-file reader and writer, workspaces and
    encryption are used, so merged catalogs are exactly what Mondocraft itself would write."""
    folder = Path(folder or DEFAULT_MONDOCRAFT).expanduser().resolve()
    if not (folder / "server.py").is_file():
        raise SystemExit(f"No Mondocraft found in {folder}. Point --mondocraft at your Mondocraft folder "
                         f"(or set MONDOCRAFT).")
    sys.path.insert(0, str(folder))
    import server
    return server


def find_workspace(server, name: str | None) -> dict:
    spaces = server.workspaces()
    if not name:
        return spaces[0]
    for ws in spaces:
        if name.lower() in (ws["id"].lower(), ws["name"].lower()):
            return ws
    raise SystemExit(f"No workspace named {name!r}. Workspaces: " + ", ".join(w["name"] for w in spaces))


def merge_files(harvests: list[Path], server, workspace: str | None = None, dry_run: bool = False,
                catalog: Path | None = None) -> dict:
    """Merge harvest files into a workspace's prompts.json (or the given catalog file) with Mondocraft's own
    prompt-file reader and writer, so an encrypted catalog stays encrypted."""
    catalog = catalog or server.workspace_paths(find_workspace(server, workspace))["prompts"]
    try:
        items = json.loads(server.read_prompt_file(catalog))
    except FileNotFoundError:
        items = []
    moved = _upgrade(items)
    total = dict(added=0, updated=0, unchanged=0)
    for path in harvests:
        counts = merge_harvest(items, json.loads(Path(path).read_text()))
        for k in total:
            total[k] += counts[k]
    if not dry_run and (total["added"] or total["updated"] or moved):
        server.write_prompt_file(catalog, json.dumps(items, indent=1))
    return {**total, "catalog": str(catalog), "prompts": len(items), "written": not dry_run}


# ---------------------------------------------------------------- command line

def _merge_command(harvests: list[Path]) -> str | None:
    """The merge command to copy after a crawl, or None when Mondocraft is not installed. Only the one folder
    DEFAULT_MONDOCRAFT names (MONDOCRAFT, else ~/mondocraft) is checked; nothing is searched for. The command uses
    Mondocraft's own Python when it is there (it has the packages Mondocraft needs), else this one."""
    mc = DEFAULT_MONDOCRAFT.resolve()
    if not (mc / "server.py").is_file():
        return None
    python = mc / ".venv" / "bin" / "python"
    parts = [python if python.exists() else Path(sys.executable), Path(__file__).resolve(), "merge",
             *(p.resolve() for p in harvests), "--mondocraft", mc]
    return " ".join(shlex.quote(str(x)) for x in parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--version", action="version", version=f"mondo-harvest {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crawl", help="harvest folders into one harvest file each")
    c.add_argument("folders", nargs="+", type=Path)
    c.add_argument("--out", type=Path, default=HARVESTS, help="where harvest files go (default: harvests/)")
    c.add_argument("--no-recurse", action="store_true", help="read only the folder itself, not its subfolders")
    c.add_argument("--keep-triggers", action="store_true",
                   help="leave LoRA trigger words at the start of prompts (by default they are taken off)")
    m = sub.add_parser("merge", help="merge harvest files into a Mondocraft workspace's saved prompts")
    m.add_argument("harvests", nargs="+", type=Path)
    m.add_argument("--mondocraft", type=Path, help=f"your Mondocraft folder (default: {DEFAULT_MONDOCRAFT})")
    m.add_argument("--workspace", help="workspace name or id (default: Main)")
    m.add_argument("--dry-run", action="store_true", help="count what would change; write nothing")
    w = sub.add_parser("workspaces", help="list Mondocraft's workspaces and their prompt files")
    w.add_argument("--mondocraft", type=Path, help=f"your Mondocraft folder (default: {DEFAULT_MONDOCRAFT})")
    a = ap.parse_args(argv)

    if a.cmd == "crawl":
        a.out.mkdir(parents=True, exist_ok=True)
        written = []
        for folder in a.folders:
            if not folder.is_dir():
                print(f"{folder}: not a folder, skipped", file=sys.stderr)
                continue
            start = time.time()
            print(f"{folder}:")
            path = harvest_path(folder, a.out)
            try:
                h = crawl(folder, recurse=not a.no_recurse, checkpoint=partial_path(path), keep_triggers=a.keep_triggers)
            except KeyboardInterrupt:
                return 130
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(h, indent=1, ensure_ascii=False))
            tmp.replace(path)
            partial_path(path).unlink(missing_ok=True)
            runs = sum(p["images"] for p in h["prompts"])
            print(f"  {h['files_scanned']} files, {h['files_with_prompts']} with prompts, {runs} prompt uses, "
                  f"{h['prompts_found']} distinct prompts, {len(h['errors'])} unreadable, "
                  f"{time.time() - start:.1f}s\n  -> {path}")
            fields = list(h["metadata_fields"].items())
            if fields:
                print("  Metadata fields (files that have it / files a prompt came from):")
                for name, n in fields[:25]:
                    print(f"    {name}: {n['files']} / {n['prompts']}")
                if len(fields) > 25:
                    print(f"    {len(fields) - 25} more are listed in the harvest file.")
            if h["lora_triggers"]:
                print("  LoRA trigger words taken off the start of prompts (prompts / trigger / LoRA; "
                      "--keep-triggers leaves them on):")
                for t in h["lora_triggers"][:25]:
                    print(f"    {t['prompts']}  {t['trigger']!r}  {t['lora']}")
                if len(h["lora_triggers"]) > 25:
                    print(f"    {len(h['lora_triggers']) - 25} more are listed in the harvest file.")
            written.append(path)
        cmd = _merge_command(written) if written else None
        if cmd:
            print(f"\nTo merge into Mondocraft's Main workspace (add --workspace NAME for another):\n"
                  f"  {cmd} --dry-run\n  {cmd}")
    elif a.cmd == "merge":
        r = merge_files(a.harvests, load_mondocraft(a.mondocraft), a.workspace, a.dry_run)
        print(f"{'Dry run: would add' if a.dry_run else 'Added'} {r['added']}, "
              f"{'would update' if a.dry_run else 'updated'} {r['updated']}, unchanged {r['unchanged']}. "
              f"{r['catalog']} {'would hold' if a.dry_run else 'holds'} {r['prompts']} prompts.")
    else:
        server = load_mondocraft(a.mondocraft)
        for ws in server.workspaces():
            print(f"{ws['name']} ({ws['id']}): {server.workspace_paths(ws)['prompts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
