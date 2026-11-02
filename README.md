# MidFR ComfyUI Nodes

Three small packs of custom [ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes: everyday workflow tools, dataset tagging, and MiniMax H3 prompt building.

> **Is this safe?** Yes. No clue why previous author said this. Probably just 'vibe-coders first code' talk, which is fair. I'm a vibe-coder too, so I get it.
Node was sent through Claude on MCP connection to test and fix. Unless bro is a genius or something... this should be safe to use.

| Pack | Name | Nodes | 
| ----- | ----- | ----- | 
| **A** | [Toolbox](A-toolbox) | 12 across 8 tools | 
| **B** | [Image Tagging](B-image-tagging) | 4 | 
| **C** | [H3 AutoPrompt](C-h3-autoprompt) | 3 | 

## Pack A: Toolbox

| Tool | Nodes | Does | 
| ----- | ----- | ----- | 
| **Resource Reaper** (`ComfyUI-Cleaner`) | Free Memory (passthrough), Free Memory After Run, Memory Status | Frees VRAM/RAM before each run; sidebar panel shows what else is hogging the GPU. | 
| **Text Edit (Pause)** (`ComfyUI-TextEditPause`) | Text Edit (Pause) | Pauses the workflow, lets you edit the text in a popup, then continues. | 
| **Image Upload + Resize** (`simple_image_upload_resize`) | Simple Image Upload + Resize (image + **mask**), Multi Image Upload + Resize (9), Image Set Splitter (9) | Drag-and-drop images, auto-resized to a megapixel cap. The mask comes from the image's alpha / the mask editor. | 
| **Saturation Match** (`comfyui-airpg-saturation-match`) | AIRPG Saturation Match | Fixes over-saturated edits by matching colour intensity to the reference image. | 
| **Strength to Denoise** (`comfyui-flow-strength`) | Strength to Denoise (flow models) | Invoke-style img2img strength converted to KSampler `denoise` (on MOSS-Image `denoise 0.9` keeps only \~2% of the source). | 
| **Image BBox Layout** (`image_bbox_layout`) | Image BBox Layout (MOSS) | Draw and prompt boxes on an image; outputs a MOSS-Image layout prompt, the boxes and the mask you paint in the editor. Unified prompt style support. | 
| **Invoke Inpainting** (`invoke_inpainting`) | Invoke Inpainting (MOSS) | InvokeAI-style box + mask inpainting in one node, plus a MOSS edit text encoder that hides the edit area from the reference. Fully updated with per-box `crop_to_box` control and robust negative prompt encoding. | 
| **Lora Vault** (`comfyui-lora-vault`) | Lora Vault Loader | LoRA manager: saves Civitai metadata, downloads weights on first use, tracks removals. | 

---

### MOSS Inpainting & BBox Layout Fixes & Troubleshooting Guide

Recent updates and bug fixes resolve several edge cases in `crop_to_box`, per-box mode, and conditioning encoding:

#### What Was Fixed
1. **Stock Encers + `crop_to_box`:** Previously, only the reference image was swapped for the crop, while vision tokens and layout box coordinates baked into the conditioning still described the whole picture (causing the model to paint an entire scene inside a small box).
2. **Per-box Mode & `crop_to_box`:** Per-box mode now explicitly honors `crop_to_box` (previously `box_working_mp` alone decided whether to crop, defaulting to always crop). **Default behavior change:** Per-box mode now edits in the whole picture unless `crop_to_box` is enabled.
3. **Prompt Styles Alignment:** A unified prompt builder now serves both paths. Layout coordinates remain relative to the target image canvas, and scene/context text are included consistently.
4. **Mush / Blocky Artifacts:** Empty negative prompts were previously replaced with all-zero conditioning vectors, which breaks MOSS at `CFG > 1`. They are now correctly encoded as empty text prompts.

#### Required Workflow Setup for Inpainting
To ensure successful object removal and inpainting without blocky artifacts or unintended scene generation:
* **Connections:** Connect `CLIP` and the BBox node's `passes` output directly to **Invoke Inpainting**.
* **Node Parameters:**
  * Turn `crop_to_box` **on**.
  * Set `context_padding` to `-1` (enlarges the crop to roughly 1 MP).
  * Set `prompt_style` to **`instruction`** when using command-style text (e.g., `"Remove the bird"`). 
  * If using **`layout`** style, describe the target content that should be in the box rather than a command.
  * *Note:* With `hide_edit_area` set to `keep`, the model copies old content, so descriptive labels like `"buildings"` will leave existing elements intact unless instructed otherwise.

---

## Pack B: Image Tagging

| Node | Does | 
| ----- | ----- | 
| **Batch Image Loader** | Loads one image per run from a folder (glob, shuffle, auto-advance). | 
| **Hybrid Tagging Node** | Makes WD14 booru tags and/or a Qwen3-VL caption, plus a combined version. | 
| **Same Directory Text Saver** | Saves the caption as a `.txt` next to the source image. | 
| **Image Batch Queue** | Counts through a batch and flags when it's finished. | 

## Pack C: H3 AutoPrompt

Writes MiniMax H3's strict structured video prompt from your reference media plus a short brief.

| Node | Does | 
| ----- | ----- | 
| **H3 AutoPrompt: Structured Media Loader** | Panel for loading and tagging pictures, video, and audio references per shot. | 
| **H3 AutoPrompt: All-in-One** | Local VLM + Whisper turn the references into a format-valid H3 prompt, with a self-check report. | 
| **H3 AutoPrompt: Reference Splitter** | Splits the bundled references into individual picture / video / audio outputs. | 

## Install

**Recommended: clone the whole repo into `custom_nodes`.** A root `__init__.py` loads every tool; if one fails (e.g. a missing dependency) it's reported in the console and the rest still load.

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/MidFR-CAI/midfr-comfyui-nodes
```

Restart ComfyUI. The console prints `[MidFR] loaded N tool(s), N node(s)`.

* **Don't also keep the old standalone copies** of these tools in `custom_nodes` (duplicate node names).
* **Skip a tool:** list its folder name (e.g. `comfyui-lora-vault`) in a `disabled.txt` next to `__init__.py`, one per line, or set `MIDFR_DISABLE=comfyui-lora-vault,ComfyUI-Cleaner`.
* **Defaults to know:** Resource Reaper frees memory before every run (turn off in its sidebar panel), and Lora Vault starts a local-only helper on `127.0.0.1`.

**Just want one tool?** Run `python install.py "C:\path\to\ComfyUI\custom_nodes" A` (pack letters A, B, C; none = all) or copy a single tool folder (e.g. `A-toolbox/ComfyUI-TextEditPause`) into `custom_nodes`. Each tool also works on its own.

Some tools have optional Python dependencies; see each pack's `requirements.txt` / README. Use ComfyUI's own Python (portable build: `python_embeded\python.exe -m pip install ...`).

## License

[MIT](LICENSE). Pack C is standalone and needs no other node pack. Its reference numbering conventions and the prompt grammar it encodes follow Adudeguyman's MIT-licensed [ComfyUI-Fantastic-MiniMaxH3-PromptBuilder](https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder) and MiniMax's H3 prompt-writing guide; thanks to both.
