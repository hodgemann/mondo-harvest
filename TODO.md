# To-do

- [ ] Treat Video Combine's `-audio` copy of a video (`clip_00001-audio.mp4` beside `clip_00001.mp4`) as the same
  output, so it is not counted twice.
- [ ] Idea under consideration: a ComfyUI custom node that works as a prompt library. It would be cabled to a prompt
  (text encode) node to supply its text, and offer a background crawl that adds newly harvested prompts to the
  library. ComfyUI asks each node for its options (`INPUT_TYPES`) when the page loads, so if the node reads the
  library file each time, a normal page reload, or ComfyUI's Refresh (the R key), shows newly added prompts without
  restarting ComfyUI. A library of thousands of prompts needs a searchable picker rather than a plain dropdown.
