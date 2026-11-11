"""mondo-harvest: reading prompts from outputs' metadata, de-duplicating them, and merging into a prompt catalog.
Builds small synthetic outputs in a temporary folder. No network; merges use an in-memory stand-in for Mondocraft.
Run: python3 -m unittest discover -s tests"""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mondo_harvest as ch  # noqa: E402


def png(path: Path, chunks: dict, size=(64, 32)) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    body = chunk(b"IHDR", struct.pack(">IIBBBBB", *size, 8, 2, 0, 0, 0))
    for k, v in chunks.items():
        body += chunk(b"tEXt", k.encode() + b"\0" + v.encode())
    body += chunk(b"IDAT", zlib.compress(b"\0" * 10)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)


def graph(text, seed, neg="blurry, extra fingers", loras=None, load_image=False, prefix="Out"):
    """A Qwen-style KSampler graph with an rgthree Power Lora Loader."""
    g = {
        "3": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 20, "cfg": 2.5, "sampler_name": "euler",
              "scheduler": "simple", "denoise": 1.0, "model": ["72", 0], "positive": ["6", 0],
              "negative": ["7", 0], "latent_image": ["58", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["72", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["38", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_vl.safetensors", "type": "qwen_image"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_vae.safetensors"}},
        "58": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 64, "height": 32, "batch_size": 1}},
        "60": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
        "72": {"class_type": "Power Lora Loader (rgthree)", "inputs": {
            "PowerLoraLoaderHeaderWidget": {"type": "PowerLoraLoaderHeaderWidget"},
            **{f"lora_{i}": l for i, l in enumerate(loras or [], 1)}, "➕ Add Lora": "",
            "model": ["37", 0], "clip": ["38", 0]}},
    }
    if load_image:
        g["76"] = {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}
        g["77"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["76", 0], "vae": ["39", 0]}}
        g["3"]["inputs"]["latent_image"] = ["77", 0]
    return g


def tiff(tags: dict, exif_ifd: dict | None = None) -> bytes:
    """A big-endian TIFF/EXIF block. Values: str (ASCII), or bytes as UNDEFINED (type 7)."""
    def ifd(entries, start, extra_link=None):
        items = sorted(entries.items()) + ([(0x8769, None)] if extra_link is not None else [])
        head = struct.pack(">H", len(items))
        data_at = start + 2 + 12 * len(items) + 4
        body, data = b"", b""
        for tag, value in items:
            if value is None:
                body += struct.pack(">HHII", tag, 4, 1, extra_link)
                continue
            raw = value.encode() + b"\0" if isinstance(value, str) else value
            typ = 2 if isinstance(value, str) else 7
            if len(raw) <= 4:
                body += struct.pack(">HHI", tag, typ, len(raw)) + raw.ljust(4, b"\0")
            else:
                body += struct.pack(">HHII", tag, typ, len(raw), data_at + len(data))
                data += raw + b"\0" * (len(raw) & 1)
        return head + body + b"\0\0\0\0" + data
    first = ifd(tags, 8, extra_link=0 if exif_ifd else None)
    if exif_ifd:
        first = ifd(tags, 8, extra_link=8 + len(first))
        return b"MM\0*" + struct.pack(">I", 8) + first + ifd(exif_ifd, 8 + len(first))
    return b"MM\0*" + struct.pack(">I", 8) + first


def webp(path: Path, exif: bytes = b"", xmp: bytes = b"") -> None:
    body = b"WEBP" + b"VP8X" + struct.pack("<I", 10) + b"\0" * 10
    for kind, data in ((b"EXIF", exif), (b"XMP ", xmp)):
        if data:
            body += kind + struct.pack("<I", len(data)) + data + b"\0" * (len(data) & 1)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


def jpeg(path: Path, exif: bytes = b"", xmp: str = "", iptc: dict | None = None, comment: str = "") -> None:
    def seg(marker, data):
        return b"\xff" + bytes([marker]) + struct.pack(">H", len(data) + 2) + data
    out = b"\xff\xd8"
    if exif:
        out += seg(0xE1, b"Exif\0\0" + exif)
    if xmp:
        out += seg(0xE1, b"http://ns.adobe.com/xap/1.0/\0" + xmp.encode())
    if iptc:
        records = b"".join(b"\x1c\x02" + bytes([ds]) + struct.pack(">H", len(v.encode())) + v.encode()
                           for ds, v in iptc.items())
        out += seg(0xED, b"Photoshop 3.0\0" + b"8BIM\x04\x04\0\0" + struct.pack(">I", len(records)) + records)
    if comment:
        out += seg(0xFE, comment.encode())
    path.write_bytes(out + b"\xff\xda\0\x02" + b"\0" * 16 + b"\xff\xd9")


XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/"
 xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/" xmp:CreatorTool="Some Image Service 2.1"
 Iptc4xmpExt:DigitalSourceType="http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia">
<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{text}</rdf:li></rdf:Alt></dc:description>
<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">Copyright 2025 somebody, all rights reserved</rdf:li></rdf:Alt></dc:rights>
<dc:subject><rdf:Bag><rdf:li>lighthouse</rdf:li><rdf:li>storm</rdf:li><rdf:li>sea</rdf:li><rdf:li>night</rdf:li></rdf:Bag></dc:subject>
</rdf:Description></rdf:RDF></x:xmpmeta>"""


def workflow(*nodes):
    return {"nodes": [dict(id=i, type=t, mode=m, widgets_values=w) for i, (t, m, w) in enumerate(nodes)]}


LORAS = [{"on": True, "lora": "mine/style.safetensors", "strength": 0.9},
         {"on": False, "lora": "mine/off.safetensors", "strength": 1.0}]


class CrawlTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def put(self, name, g, wf=None, size=(64, 32)):
        png(self.dir / name, {"prompt": json.dumps(g), **({"workflow": json.dumps(wf)} if wf else {})}, size)

    def test_repeats_of_a_prompt_are_one_record_with_every_seed(self):
        for i, seed in enumerate((101, 102, 103), 1):
            self.put(f"Out_{i:05}_.png", graph("a red  fox,\nsnow" if i == 2 else "a red fox, snow", seed, loras=LORAS))
        h = ch.crawl(self.dir)
        self.assertEqual(h["prompts_found"], 1)
        p = h["prompts"][0]
        self.assertEqual((p["kind"], p["t2i"], p["images"]), ("image", True, 3))
        [setup] = p["setups"]
        self.assertEqual([r["seed"] for r in setup["runs"]], [101, 102, 103])
        self.assertEqual(setup["models"], ["qwen_image.safetensors"])
        self.assertEqual(setup["text_encoders"], ["qwen_vl.safetensors"])
        self.assertEqual(setup["loras"], [{"name": "mine/style.safetensors", "strength": 0.9}])   # switched-off one left out
        self.assertEqual(setup["negative"], "blurry, extra fingers")
        self.assertEqual((setup["width"], setup["height"], setup["steps"], setup["cfg"]), (64, 32, 20, 2.5))

    def test_a_different_setup_is_a_second_group_under_the_same_prompt(self):
        self.put("Out_00001_.png", graph("a red fox", 1))
        self.put("Out_00002_.png", graph("a red fox", 2), size=(128, 64))
        self.put("Out_00003_.png", graph("a red fox", 3, loras=LORAS))
        [p] = ch.crawl(self.dir)["prompts"]
        self.assertEqual(len(p["setups"]), 3)

    def test_unused_workflow_text_is_kept_but_not_negatives_or_notes(self):
        wf = workflow(("CLIPTextEncode", 0, ["a red fox"]),
                      ("CLIPTextEncode", 4, ["an old blue heron at dawn"]),       # bypassed: a prompt from before
                      ("ShowText|pysssss", 0, [["the previous prompt shown here"]]),
                      ("CLIPTextEncode", 2, ["blurry, extra fingers"]),         # muted copy of the negative
                      ("Note", 0, ["remember to raise the shift value"]),
                      ("CR SDXL Aspect Ratio", 0, [1024, 1024, "9:16 portrait 768x1344"]))
        self.put("Out_00001_.png", graph("a red fox", 7), wf)
        h = ch.crawl(self.dir)
        by_text = {p["text"]: p for p in h["prompts"]}
        self.assertEqual(set(by_text), {"a red fox", "an old blue heron at dawn", "the previous prompt shown here"})
        self.assertEqual(by_text["an old blue heron at dawn"]["unused_in"], ["Out_00001_.png"])
        self.assertEqual(by_text["an old blue heron at dawn"]["images"], 0)
        self.assertNotIn("unused_in", by_text["a red fox"])

    def test_reference_image_upstream_makes_it_not_text_to_image(self):
        self.put("Out_00001_.png", graph("a red fox", 1, load_image=True))
        self.assertFalse(ch.crawl(self.dir)["prompts"][0]["t2i"])

    def test_unused_load_image_does_not_count(self):
        g = graph("a red fox", 1)
        g["76"] = {"class_type": "LoadImage", "inputs": {"image": "ref.png"}}
        self.put("Out_00001_.png", g)
        self.assertTrue(ch.crawl(self.dir)["prompts"][0]["t2i"])

    def test_file_comes_from_the_save_node_with_the_longest_matching_prefix(self):
        g = graph("first pass prompt", 5)
        g["90"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "upscale pass prompt", "clip": ["38", 0]}}
        g["91"] = {"class_type": "KSampler", "inputs": {**g["3"]["inputs"], "seed": 9, "positive": ["90", 0],
                                                        "latent_image": ["3", 0]}}
        g["92"] = {"class_type": "VAEDecode", "inputs": {"samples": ["91", 0], "vae": ["39", 0]}}
        g["93"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "OutBIG", "images": ["92", 0]}}
        self.put("OutBIG_00001_.png", g)
        self.put("Out_00001_.png", g)
        h = ch.crawl(self.dir)
        by_text = {p["text"]: p for p in h["prompts"]}
        self.assertEqual(by_text["upscale pass prompt"]["setups"][0]["runs"][0]["file"], "OutBIG_00001_.png")
        big = by_text["first pass prompt"]["setups"]
        files = sorted(r["file"] for s in big for r in s["runs"])
        self.assertEqual(files, ["OutBIG_00001_.png", "Out_00001_.png"])
        two_pass = next(s for s in big if s.get("passes") == 2)
        self.assertEqual((two_pass["runs"][0]["seed"], two_pass["runs"][0]["more_seeds"]), (5, [9]))

    def test_flux_custom_sampler_seed_guidance_and_lora_stack(self):
        g = {
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "neon alley at night", "clip": ["11", 0]}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["10", 0]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "sub/Flux", "images": ["8", 0]}},
            "10": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "11": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5.safetensors", "clip_name2": "l.safetensors"}},
            "12": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux.safetensors"}},
            "13": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["25", 0], "guider": ["22", 0],
                   "sampler": ["16", 0], "sigmas": ["17", 0], "latent_image": ["85", 0]}},
            "16": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "dpmpp_2m"}},
            "17": {"class_type": "BasicScheduler", "inputs": {"scheduler": "sgm_uniform", "steps": 30, "denoise": 1.0,
                   "model": ["30", 0]}},
            "22": {"class_type": "BasicGuider", "inputs": {"model": ["30", 0], "conditioning": ["60", 0]}},
            "25": {"class_type": "RandomNoise", "inputs": {"noise_seed": 4242}},
            "30": {"class_type": "Lora Loader Stack (rgthree)", "inputs": {"lora_01": "a.safetensors", "strength_01": 0.5,
                   "lora_02": "None", "strength_02": 1.0, "lora_03": "b.safetensors", "strength_03": 0,
                   "model": ["12", 0], "clip": ["11", 0]}},
            "60": {"class_type": "FluxGuidance", "inputs": {"guidance": 3.5, "conditioning": ["6", 0]}},
            "85": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 32, "batch_size": 1}},
        }
        self.put("Flux_00001_.png", g)
        [p] = ch.crawl(self.dir)["prompts"]
        [s] = p["setups"]
        self.assertEqual(s["runs"][0]["seed"], 4242)
        self.assertEqual((s["sampler"], s["scheduler"], s["steps"], s["guidance"]), ("dpmpp_2m", "sgm_uniform", 30, 3.5))
        self.assertEqual(s["loras"], [{"name": "a.safetensors", "strength": 0.5}])      # strength 0 and None left out
        self.assertEqual(s["models"], ["flux.safetensors"])
        self.assertNotIn("negative", s)

    def test_webp_exif(self):
        exif = tiff({0x010F: "workflow:" + json.dumps(workflow(("CLIPTextEncode", 4, ["older webp prompt text"]))),
                     0x0110: "prompt:" + json.dumps(graph("a webp fox", 3))})                # how ComfyUI writes WebP
        webp(self.dir / "Out_00001_.webp", exif=exif)
        texts = {p["text"] for p in ch.crawl(self.dir)["prompts"]}
        self.assertEqual(texts, {"a webp fox", "older webp prompt text"})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_video_tags_and_its_first_frame_png_count_once(self):
        g = graph("a slow pan over a harbour", 77, prefix="Clip")
        g["60"] = {"class_type": "VHS_VideoCombine", "inputs": {"filename_prefix": "Clip", "images": ["8", 0]}}
        comment = json.dumps({"prompt": json.dumps(g), "workflow": {"nodes": []}})
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x16:d=0.1", "-metadata",
                        f"comment={comment}", "-pix_fmt", "yuv420p", str(self.dir / "Clip_00001.mp4")], check=True)
        self.put("Clip_00001.png", g)
        h = ch.crawl(self.dir)
        [p] = h["prompts"]
        self.assertEqual((p["kind"], p["images"]), ("video", 1))
        run = p["setups"][0]["runs"][0]
        self.assertEqual((run["file"], run["seed"]), ("Clip_00001.mp4", 77))
        self.assertEqual((p["setups"][0]["width"], p["setups"][0]["height"]), (32, 16))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_video_combine_audio_copy_counts_once(self):
        g = graph("waves on a pebble beach", 78, prefix="Clip")
        g["60"] = {"class_type": "VHS_VideoCombine", "inputs": {"filename_prefix": "Clip", "images": ["8", 0]}}
        comment = json.dumps({"prompt": json.dumps(g), "workflow": {"nodes": []}})
        for name in ("Clip_00001.mp4", "Clip_00001-audio.mp4"):
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x16:d=0.1", "-metadata",
                            f"comment={comment}", "-pix_fmt", "yuv420p", str(self.dir / name)], check=True)
        self.put("Clip_00001.png", g)
        h = ch.crawl(self.dir)
        [p] = h["prompts"]
        self.assertEqual(p["images"], 1)
        self.assertEqual(p["setups"][0]["runs"][0]["file"], "Clip_00001.mp4")
        self.assertEqual((h["files_with_prompts"], h["files_without_prompts"]), (1, []))

    def test_audio_copy_stem_and_order(self):
        d = Path("/x")
        self.assertEqual(ch._output_stem(d / "a" / "clip_00001-audio.mp4", d), "a/clip_00001")
        self.assertEqual(ch._output_stem(d / "still-audio.png", d), "still-audio")    # only videos have copies

    def test_progress_from_before_keeps_the_audio_copy_as_read(self):
        clip = self.dir / "clip_00001-audio.mp4"
        clip.write_bytes(b"x")
        cp = self.dir / "progress.json"
        cp.write_text(json.dumps(dict(format=ch.PARTIAL_FORMAT, folder=str(self.dir.resolve()), recursive=True,
                                      seen={clip.name: ch._file_id(clip)}, records=[], unread=[],
                                      done_stems=["clip_00001-audio"], negatives=[], fields={})))
        state = ch._load_partial(cp, self.dir.resolve(), True, [clip.resolve()], lambda *_: None)
        self.assertIn("clip_00001", state["done_stems"])

    def test_files_without_data_and_broken_files_are_listed_not_fatal(self):
        png(self.dir / "plain.png", {})
        (self.dir / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n\0\0\0\x0dIHDR")
        self.put("Out_00001_.png", graph("a red fox", 1))
        h = ch.crawl(self.dir)
        self.assertEqual(h["prompts_found"], 1)
        self.assertEqual(sorted(h["files_without_prompts"]), ["broken.png", "plain.png"])

    def test_subfolders_are_crawled_unless_told_not_to(self):
        (self.dir / "deeper").mkdir()
        png(self.dir / "deeper" / "Out_00001_.png", {"prompt": json.dumps(graph("a deep fox", 1))})
        self.assertEqual(ch.crawl(self.dir)["prompts_found"], 1)
        self.assertEqual(ch.crawl(self.dir, recurse=False)["prompts_found"], 0)


class OtherMetadataTests(unittest.TestCase):
    """Prompts outside ComfyUI: description and comment fields, JSON fields, and settings text."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def crawl(self):
        h = ch.crawl(self.dir)
        return h, {p["text"]: p for p in h["prompts"]}

    def test_jpeg_description_fields_and_junk_left_out(self):
        jpeg(self.dir / "a.jpg",
             exif=tiff({0x010E: "a lighthouse in a storm at night", 0x010F: "Canon", 0x0131: "Photo Editor 3.2 build 17",
                        0x8298: "Copyright 2025 somebody, all rights reserved"},
                       exif_ifd={0x9286: b"ASCII\0\0\0a lighthouse in a storm at night"}),
             xmp=XMP.format(text="a lighthouse in a storm at night"),
             iptc={120: "a lighthouse in a storm at night", 116: "Copyright someone", 25: "lighthouse"})
        h, by_text = self.crawl()
        self.assertEqual(list(by_text), ["a lighthouse in a storm at night"])
        [setup] = by_text["a lighthouse in a storm at night"]["setups"]
        self.assertEqual(setup["source"], "exif:ImageDescription")   # the first field that held it; the rest are copies
        self.assertNotIn("t2i", by_text["a lighthouse in a storm at night"])   # unknown outside ComfyUI
        f = h["metadata_fields"]
        self.assertEqual(f["exif:ImageDescription"], {"files": 1, "prompts": 1})
        for name in ("exif:Make", "exif:Software", "exif:Copyright", "exif:UserComment", "xmp:dc:description",
                     "xmp:dc:rights", "xmp:dc:subject", "xmp:xmp:CreatorTool", "iptc:Caption", "iptc:Copyright",
                     "iptc:Keywords"):
            self.assertIn(name, f)
        self.assertEqual(f["xmp:dc:subject"]["prompts"], 0)

    def test_photos_and_scans_give_no_prompts_unless_marked_as_ai(self):
        museum = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/"
 xmlns:mus="http://example.org/collection/1.0/" xmpRights:Marked="True" mus:ObjectNumber="12.34">
<dc:description><rdf:Alt><rdf:li xml:lang="x-default">Bronze statuette of a dancer, cast in two parts, Roman, 2nd century</rdf:li></rdf:Alt></dc:description>
<mus:Title>Statuette of a dancing satyr with cymbals</mus:Title>
</rdf:Description></rdf:RDF></x:xmpmeta>"""
        jpeg(self.dir / "scan.jpg", xmp=museum, iptc={120: "Bronze statuette of a dancer, cast in two parts", 80: "A Photographer"},
             comment="Gift of a generous donor in memory of a collector, 1917")
        jpeg(self.dir / "photo.jpg", exif=tiff({0x010E: "sunset over the harbour with fishing boats", 0x010F: "Nikon"}))
        jpeg(self.dir / "ai.jpg", exif=tiff({0x010E: "a bronze robot dancing in a museum hall", 0x8298: "Some AI Service"}))
        h, by_text = self.crawl()
        self.assertEqual(list(by_text), [])
        self.assertIn("xmp:collection:ObjectNumber", h["metadata_fields"])        # unknown namespace named, version skipped
        data = (self.dir / "ai.jpg").read_bytes()
        (self.dir / "ai.jpg").write_bytes(data[:2] + b"\xff\xeb\0\x0eJP\0\x01\0\0\0\0jumb" + data[2:])   # a C2PA block
        _, by_text = self.crawl()
        self.assertEqual(list(by_text), ["a bronze robot dancing in a museum hall"])

    def test_titles_are_not_prompts_but_custom_prompt_fields_are(self):
        custom = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:gen="http://example.com/genservice/2.0/" gen:Prompt="a paper boat sailing down a rainy street"
 gen:Notes="rendered on the fast queue with default settings"/></rdf:RDF></x:xmpmeta>"""
        webp(self.dir / "a.webp", xmp=custom.encode())
        png(self.dir / "b.png", {"Title": "Episode twelve of the long running space series"})
        _, by_text = self.crawl()
        self.assertEqual(list(by_text), ["a paper boat sailing down a rainy street"])
        self.assertEqual(by_text["a paper boat sailing down a rainy street"]["setups"][0]["source"], "xmp:genservice:Prompt")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_video_title_is_not_a_prompt(self):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x16:d=0.1", "-metadata",
                        "title=Space Series (S06E01) The Long Way Home", "-pix_fmt", "yuv420p",
                        str(self.dir / "ep.mp4")], check=True)
        h, by_text = self.crawl()
        self.assertEqual((by_text, h["metadata_fields"]["video:title"]["prompts"]), ({}, 0))

    def test_xmp_in_webp_and_png(self):
        webp(self.dir / "a.webp", xmp=XMP.format(text="a red kite over the moors").encode())
        h, by_text = self.crawl()
        self.assertEqual(by_text["a red kite over the moors"]["setups"][0]["source"], "xmp:dc:description")

    def test_short_labels_links_and_credit_lines_are_not_prompts(self):
        for i, text in enumerate(("AI generated image", "https://example.com/some/image/page", "Made with Some AI tool",
                                  "untitled image number four", "0f8fad5b-d9cb-469f-a165-70867728950e")):
            png(self.dir / f"{i}.png", {"Description": text})
        h, by_text = self.crawl()
        self.assertEqual(by_text, {})
        self.assertEqual(h["metadata_fields"]["png:Description"], {"files": 5, "prompts": 0})
        self.assertEqual(len(h["files_without_prompts"]), 5)

    def test_settings_text_is_split(self):
        png(self.dir / "a.png", {"parameters": "a castle on a hill, <lora:mystyle:0.7>\nNegative prompt: blurry, "
                                               "low quality\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 7.5, "
                                               "Seed: 12345, Size: 512x768, Model: dreamy_v2"})
        _, by_text = self.crawl()
        [s] = by_text["a castle on a hill"]["setups"]
        self.assertEqual((s["negative"], s["steps"], s["sampler"], s["cfg"], s["models"], s["runs"][0]["seed"]),
                         ("blurry, low quality", 30, "DPM++ 2M", 7.5, ["dreamy_v2"], 12345))

    def test_json_comment_field(self):
        png(self.dir / "a.png", {"Comment": json.dumps({"prompt": "a quiet harbour at dawn, oil painting",
                                                        "uc": "text, watermark", "seed": 99, "steps": 28}),
                                 "Title": "AI generated image", "Software": "Some Service"})
        _, by_text = self.crawl()
        [s] = by_text["a quiet harbour at dawn, oil painting"]["setups"]
        self.assertEqual((s["source"], s["negative"], s["runs"][0]["seed"]), ("png:Comment", "text, watermark", 99))

    def test_trailing_job_id_is_dropped(self):
        jpeg(self.dir / "a.jpg", xmp=XMP.format(
            text="a fox in tall grass --ar 16:9 Job ID: 0f8fad5b-d9cb-469f-a165-70867728950e"))
        _, by_text = self.crawl()
        self.assertEqual(list(by_text), ["a fox in tall grass --ar 16:9"])

    def test_png_prompt_field_that_is_not_a_comfyui_graph(self):
        png(self.dir / "a.png", {"prompt": "a glass teapot full of stars"})
        _, by_text = self.crawl()
        self.assertEqual(by_text["a glass teapot full of stars"]["setups"][0]["source"], "png:prompt")

    def test_same_prompt_from_comfyui_and_a_service_is_one_record(self):
        png(self.dir / "Out_00001_.png", {"prompt": json.dumps(graph("a red fox in the snow", 1))})
        jpeg(self.dir / "b.jpg", exif=tiff({0x010E: "a red fox in the snow"}))
        _, by_text = self.crawl()
        self.assertEqual(sorted(s["source"] for s in by_text["a red fox in the snow"]["setups"]),
                         ["comfyui", "exif:ImageDescription"])
        self.assertTrue(by_text["a red fox in the snow"]["t2i"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_video_description_tag(self):
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x16:d=0.1", "-metadata",
                        "description=a slow dolly shot through a flooded library", "-metadata", "title=clip",
                        "-pix_fmt", "yuv420p", str(self.dir / "v.mp4")], check=True)
        _, by_text = self.crawl()
        p = by_text["a slow dolly shot through a flooded library"]
        self.assertEqual((p["kind"], p["setups"][0]["source"]), ("video", "video:description"))


OHWX = [{"on": True, "lora": "chars/ohwx_woman_v2.safetensors", "strength": 0.8},
        {"on": True, "lora": "Mjolnir_Style-FLUX.safetensors", "strength": 1.0}]


class TriggerTests(unittest.TestCase):
    """LoRA trigger words at the start of a prompt are taken off and kept with the LoRA."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)

    def put(self, name, g, wf=None):
        png(self.dir / name, {"prompt": json.dumps(g), **({"workflow": json.dumps(wf)} if wf else {})})

    def strip(self, text, *names):
        return ch.strip_triggers(text, [{"name": n, "strength": 1} for n in names])

    def test_trigger_spelled_by_the_lora_name_is_taken_off(self):
        self.assertEqual(self.strip("ohwx woman, a portrait in the rain", "chars/ohwx_woman_v2.safetensors"),
                         ("a portrait in the rain", {"chars/ohwx_woman_v2.safetensors": "ohwx woman"}))
        self.assertEqual(self.strip("pixelart. a castle on a hill", "pixel-art-xl-v1.1.safetensors")[0],
                         "a castle on a hill")
        self.assertEqual(self.strip("(ohwx:1.2), a man walking a dog", "flux\\OhwxMan_e10.safetensors")[0],
                         "a man walking a dog")
        self.assertEqual(self.strip("ohwx woman, Mjolnir style, a woman holding a hammer",
                                    "ohwx_woman.safetensors", "Mjolnir_Style-FLUX.safetensors"),
                         ("a woman holding a hammer", {"ohwx_woman.safetensors": "ohwx woman",
                                                       "Mjolnir_Style-FLUX.safetensors": "Mjolnir style"}))

    def test_text_that_is_not_a_trigger_stays(self):
        for text, name in (("a woman, red dress, in the rain", "woman_detailer.safetensors"),   # not the name
                           ("style, a cat sitting on a mat", "anime_style.safetensors"),         # name noise only
                           ("flux, a cat sleeping on a sofa", "cat-FLUX.safetensors"),
                           ("art, a smart robot painting", "smart_robot.safetensors"),          # inside a word
                           ("ohwx woman wearing a red dress", "ohwx_woman.safetensors"),        # no separator
                           ("ohwx woman, cat", "ohwx_woman.safetensors"),                       # too little left
                           ("a cat. ohwx, on a mat", "ohwx.safetensors")):                      # not at the start
            self.assertEqual(self.strip(text, name), (text, {}), text)

    def test_lora_tags_are_taken_off_with_their_separator(self):
        for text, want in (("<lora:Hodge/H3rdBW.safetensors:0.93>, a woman in the rain", "a woman in the rain"),
                           ("a castle, <lora:style:0.7>, at night", "a castle, at night"),
                           ("a castle on a hill, <lora:style:0.7>", "a castle on a hill"),
                           ("a castle <LORA:style:1> on a hill", "a castle on a hill"),
                           ("<lora:a:1><lora:b:0.5> a cat. <lora:c:1>. on a mat", "a cat. on a mat"),
                           ("<lora:a:1>\na cat on a mat", "a cat on a mat"),
                           ("a cat on a mat, <lorax> is not closed <lora", "a cat on a mat, is not closed <lora"),
                           ("a cat on a mat", "a cat on a mat")):
            self.assertEqual(ch.strip_lora_tags(text), want, text)

    def test_crawl_takes_lora_tags_off_before_triggers(self):
        self.put("Out_00001_.png", graph("<lora:Hodge/H3rdBW.safetensors:0.93>, ohwx woman, a portrait in the rain",
                                         1, loras=OHWX))
        self.put("Out_00002_.png", graph("<lora:x:1>", 2))
        h = ch.crawl(self.dir, log=lambda *_: None)
        self.assertEqual([p["text"] for p in h["prompts"]], ["a portrait in the rain"])

    def test_crawl_keeps_the_trigger_with_the_lora_and_joins_the_clean_prompt(self):
        self.put("Out_00001_.png", graph("ohwx woman, a portrait in the rain", 1, loras=OHWX))
        self.put("Out_00002_.png", graph("a portrait in the rain", 2, loras=OHWX))
        h = ch.crawl(self.dir, log=lambda *_: None)
        [p] = h["prompts"]
        self.assertEqual(p["text"], "a portrait in the rain")
        with_trigger, without = p["setups"]
        self.assertEqual(with_trigger["loras"][0], {"name": "chars/ohwx_woman_v2.safetensors", "strength": 0.8,
                                                    "trigger": "ohwx woman"})
        self.assertNotIn("trigger", without["loras"][0])
        self.assertEqual(h["lora_triggers"], [{"lora": "chars/ohwx_woman_v2.safetensors", "trigger": "ohwx woman",
                                               "prompts": 1}])

    def test_unused_workflow_text_loses_its_trigger_too(self):
        wf = {"nodes": [{"type": "CLIPTextEncode", "mode": 4, "widgets_values": ["Mjolnir style, a hammer on an anvil"]},
                        {"type": "CLIPTextEncode", "widgets_values": ["ohwx woman, a portrait in the rain"]}]}
        self.put("Out_00001_.png", graph("ohwx woman, a portrait in the rain", 1, loras=OHWX), wf)
        by_text = {p["text"]: p for p in ch.crawl(self.dir, log=lambda *_: None)["prompts"]}
        self.assertEqual(set(by_text), {"a portrait in the rain", "a hammer on an anvil"})
        self.assertNotIn("unused_in", by_text["a portrait in the rain"])
        self.assertEqual(by_text["a hammer on an anvil"]["unused_in"], ["Out_00001_.png"])

    def test_keep_triggers_and_lora_switched_off(self):
        self.put("Out_00001_.png", graph("ohwx woman, a portrait in the rain", 1, loras=OHWX))
        self.put("Out_00002_.png", graph("ohwx woman, a dancer on a stage", 2, loras=[{**OHWX[0], "on": False}]))
        texts = {p["text"] for p in ch.crawl(self.dir, keep_triggers=True, log=lambda *_: None)["prompts"]}
        self.assertEqual(texts, {"ohwx woman, a portrait in the rain", "ohwx woman, a dancer on a stage"})
        texts = {p["text"] for p in ch.crawl(self.dir, log=lambda *_: None)["prompts"]}
        self.assertEqual(texts, {"a portrait in the rain", "ohwx woman, a dancer on a stage"})

    def test_progress_saved_with_the_other_setting_is_not_used(self):
        self.put("Out_00001_.png", graph("ohwx woman, a portrait in the rain", 1, loras=OHWX))
        cp = self.dir.parent / (self.dir.name + ".partial")
        self.addCleanup(cp.unlink, missing_ok=True)
        cp.write_text(json.dumps(dict(format=ch.PARTIAL_FORMAT, folder=str(self.dir.resolve()), recursive=True,
                                      seen={"Out_00001_.png": ch._file_id(self.dir / "Out_00001_.png")},
                                      records=[], unread=[], done_stems=[], negatives=[], fields={})))
        h = ch.crawl(self.dir, checkpoint=cp, log=lambda *_: None)                # 1.0.0 progress kept triggers
        self.assertEqual([p["text"] for p in h["prompts"]], ["a portrait in the rain"])


class ResumeTests(unittest.TestCase):
    """An interrupted crawl saves its progress and a later crawl of the same folder carries on from it."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.src = self.dir / "src"
        self.src.mkdir()
        for i in range(1, 7):
            png(self.src / f"Out_{i:05}_.png", {"prompt": json.dumps(graph(f"prompt number {i % 3}", i))})
        self.cp = self.dir / "src.json.partial"
        self.full = ch.crawl(self.src)

    def interrupted(self, after):
        """Crawl, saving at every file, and stop with Ctrl+C when the given file number is reached."""
        real, count = ch.read_fields, [0]

        def reading(path):
            count[0] += 1
            if count[0] == after:
                raise KeyboardInterrupt
            return real(path)
        with patch.object(ch, "read_fields", reading), self.assertRaises(KeyboardInterrupt):
            ch.crawl(self.src, checkpoint=self.cp, save_every=0, log=lambda *_: None)

    def resumed(self):
        real, read = ch.read_fields, []

        def reading(path):
            read.append(path.name)
            return real(path)
        with patch.object(ch, "read_fields", reading):
            h = ch.crawl(self.src, checkpoint=self.cp, log=lambda *_: None)
        return h, read

    def same(self, a, b):
        drop = lambda h: {k: v for k, v in h.items() if k != "harvested_at"}
        self.assertEqual(drop(a), drop(b))

    def test_carries_on_where_it_stopped_and_gives_the_same_harvest(self):
        self.interrupted(after=4)
        state = json.loads(self.cp.read_text())
        self.assertEqual(len(state["seen"]), 3)
        h, read = self.resumed()
        self.assertEqual(read, ["Out_00004_.png", "Out_00005_.png", "Out_00006_.png"])
        self.same(h, self.full)

    def test_a_changed_file_starts_the_crawl_over(self):
        self.interrupted(after=4)
        png(self.src / "Out_00001_.png", {"prompt": json.dumps(graph("a changed prompt", 1))})
        h, read = self.resumed()
        self.assertEqual(len(read), 6)
        self.assertIn("a changed prompt", {p["text"] for p in h["prompts"]})

    def test_files_that_failed_are_read_again(self):
        real = ch.read_fields

        def flaky(path):
            if path.name == "Out_00002_.png":
                raise OSError("drive asleep")
            if path.name == "Out_00005_.png":
                raise KeyboardInterrupt
            return real(path)
        with patch.object(ch, "read_fields", flaky), self.assertRaises(KeyboardInterrupt):
            ch.crawl(self.src, checkpoint=self.cp, save_every=0, log=lambda *_: None)
        h, read = self.resumed()
        self.assertEqual(read, ["Out_00002_.png", "Out_00005_.png", "Out_00006_.png"])
        self.same(h, self.full)

    def test_progress_of_another_folder_is_not_used(self):
        self.interrupted(after=4)
        state = json.loads(self.cp.read_text())
        self.cp.write_text(json.dumps({**state, "folder": "/somewhere/else"}))
        _, read = self.resumed()
        self.assertEqual(len(read), 6)

    def test_merge_refuses_progress_files_and_names_avoid_them(self):
        self.interrupted(after=4)
        with self.assertRaises(ValueError):
            ch.merge_harvest([], json.loads(self.cp.read_text()))
        self.assertEqual(ch.harvest_path(self.src, self.dir).name, "src.json")
        self.assertEqual(ch.partial_path(self.dir / "src.json"), self.cp)
        self.assertRegex(ch.harvest_path(Path("/other/src"), self.dir).name, r"^src-[0-9a-f]{6}\.json$")

    def test_command_line_removes_the_progress_file_when_done(self):
        out = self.dir / "out"
        with patch("builtins.print"):
            ch.main(["crawl", str(self.src), "--out", str(out)])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["src.json"])

    def test_merge_command_printed_only_when_mondocraft_is_installed(self):
        out, mc = self.dir / "out", self.dir / "mondocraft"
        mc.mkdir()
        for installed in (False, True):
            if installed:
                (mc / "server.py").write_text("")
            with patch.object(ch, "DEFAULT_MONDOCRAFT", mc), patch("builtins.print") as printed:
                ch.main(["crawl", str(self.src), "--out", str(out)])
            text = "\n".join(str(c.args[0]) for c in printed.call_args_list if c.args)
            self.assertEqual(" merge " in text, installed)
            self.assertNotIn("MONDOCRAFT_FOLDER", text)


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        for i, seed in enumerate((1, 2), 1):
            png(self.dir / f"Out_{i:05}_.png", {"prompt": json.dumps(graph("a red fox, snow", seed, loras=LORAS)),
                                                 "workflow": json.dumps(workflow(("CLIPTextEncode", 4, ["an old heron prompt"])))})
        self.harvest = ch.crawl(self.dir)

    def test_adds_new_prompts_once(self):
        items = []
        self.assertEqual(ch.merge_harvest(items, self.harvest), {"added": 2, "updated": 0, "unchanged": 0})
        self.assertEqual(ch.merge_harvest(items, self.harvest), {"added": 0, "updated": 0, "unchanged": 2})
        fox = next(p for p in items if p["text"] == "a red fox, snow")
        self.assertEqual((fox["model"], fox["kind"], fox["t2i"]), ("", "image", True))
        self.assertEqual(fox["harvest"]["setups"][0]["seeds"], [1, 2])
        self.assertNotIn("runs", fox["harvest"]["setups"][0])
        self.assertEqual(fox["harvest"]["setups"][0]["loras"], [{"name": "mine/style.safetensors", "strength": 0.9}])
        heron = next(p for p in items if p["text"] == "an old heron prompt")
        self.assertEqual((heron["harvest"]["setups"], heron["harvest"]["unused_in_workflows"]), ([], 2))

    def test_existing_saved_prompt_gains_the_data_and_is_not_duplicated(self):
        items = [dict(id="abc", text="a red  fox,\nsnow", model="gpt2", kind="image", created=5, last_used=9e9)]
        counts = ch.merge_harvest(items, self.harvest)
        self.assertEqual((counts["added"], counts["updated"]), (1, 1))
        own = items[0]
        self.assertEqual((own["id"], own["text"], own["model"], own["last_used"], own["created"]),
                         ("abc", "a red  fox,\nsnow", "gpt2", 9e9, 5))
        self.assertEqual(own["harvest"]["setups"][0]["seeds"], [1, 2])

    def test_same_text_as_a_video_prompt_is_separate(self):
        items = [dict(id="v", text="a red fox, snow", model="", kind="video", created=1, last_used=1)]
        ch.merge_harvest(items, self.harvest)
        self.assertEqual(sorted(p["kind"] for p in items if p["text"] == "a red fox, snow"), ["image", "video"])

    def test_new_seeds_from_another_folder_join_the_same_setup(self):
        items = []
        ch.merge_harvest(items, self.harvest)
        other = json.loads(json.dumps(self.harvest))
        other["folder"] = "/elsewhere"
        for p in other["prompts"]:
            for s in p["setups"]:
                for r in s["runs"]:
                    r["seed"] += 100
        ch.merge_harvest(items, other)
        fox = next(p for p in items if p["text"] == "a red fox, snow")
        self.assertEqual(fox["harvest"]["setups"][0]["seeds"], [1, 2, 101, 102])
        self.assertEqual(fox["harvest"]["folders"], [str(self.dir.resolve()), "/elsewhere"])

    def test_rejects_other_json(self):
        with self.assertRaises(ValueError):
            ch.merge_harvest([], {"prompts": []})

    def test_old_harvests_and_old_catalog_entries_carry_on(self):
        old = json.loads(json.dumps(self.harvest))
        old["format"] = "mondocraft-comfyui-harvest/1"
        for p in old["prompts"]:
            for s in p["setups"]:
                s.pop("source")
        items = [dict(id="a", text="a red fox, snow", model="", kind="image", created=1, last_used=1, t2i=True,
                      comfyui={"folders": [str(self.dir.resolve())], "setups": [
                          {k: v for k, v in ch._catalog_setup(old["prompts"][0]["setups"][0]).items() if k != "source"}]})]
        ch.merge_harvest(items, old)
        fox = items[0]
        self.assertNotIn("comfyui", fox)
        self.assertEqual(len(fox["harvest"]["setups"]), 1)                      # recognised as the same setup
        self.assertEqual(fox["harvest"]["setups"][0]["source"], "comfyui")

    def test_t2i_left_out_when_unknown(self):
        h = json.loads(json.dumps(self.harvest))
        for p in h["prompts"]:
            p.pop("t2i", None)
        items = []
        ch.merge_harvest(items, h)
        self.assertTrue(all("t2i" not in p for p in items))

    def test_merge_files_writes_through_mondocraft_prompt_file_functions(self):
        hfile, catalog = self.dir / "h.json", self.dir / "prompts.json"
        hfile.write_text(json.dumps(self.harvest))
        catalog.write_text(json.dumps([dict(id="x", text="kept", model="", kind="image", created=1, last_used=1)]))
        server = FakeMondocraft(self.dir)
        r = ch.merge_files([hfile], server, dry_run=True)
        self.assertEqual((r["added"], r["prompts"]), (2, 3))
        self.assertEqual(server.writes, 0)                                       # dry run wrote nothing
        ch.merge_files([hfile], server, workspace="test")                       # no catalog yet: one is made
        self.assertEqual(len(json.loads((self.dir / "test_prompts.json").read_text())), 2)
        ch.merge_files([hfile], server)
        self.assertEqual(server.writes, 2)
        ch.merge_files([hfile], server)
        self.assertEqual(server.writes, 2)                                       # nothing new: not written again
        self.assertEqual([p["text"] for p in json.loads(catalog.read_text())][0], "kept")
        self.assertEqual(len(json.loads(catalog.read_text())), 3)
        with self.assertRaises(SystemExit):
            ch.merge_files([hfile], server, workspace="nope")

    def test_harvest_file_name_per_folder(self):
        out = self.dir / "out"
        out.mkdir()
        a = ch.harvest_path(Path("/x/batch"), out)
        self.assertEqual(a.name, "batch.json")
        a.write_text(json.dumps({"folder": "/x/batch"}))
        self.assertEqual(ch.harvest_path(Path("/x/batch"), out).name, "batch.json")
        self.assertRegex(ch.harvest_path(Path("/y/batch"), out).name, r"^batch-[0-9a-f]{6}\.json$")


class FakeMondocraft:
    """Stands in for Mondocraft's server module: the four functions merge uses, on plain files in a folder."""

    def __init__(self, folder):
        self.folder, self.writes = folder, 0

    def workspaces(self):
        return [dict(id="main", name="Main", main=True), dict(id="test", name="Test", main=False)]

    def workspace_paths(self, ws):
        return {"prompts": self.folder / ("prompts.json" if ws["main"] else "test_prompts.json")}

    def read_prompt_file(self, path):
        return Path(path).read_text()

    def write_prompt_file(self, path, text):
        self.writes += 1
        Path(path).write_text(text)


class VersionTest(unittest.TestCase):
    def test_version_file_matches_module(self):
        self.assertEqual((ROOT / "VERSION").read_text().strip(), ch.__version__)


if __name__ == "__main__":
    unittest.main()
