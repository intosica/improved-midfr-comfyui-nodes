"""FLUX.2 versions of the inpainting node and the edit encoders.

FLUX.2 (Klein / dev) is told what to copy through REFERENCE LATENTS only (ComfyUI's ReferenceLatent node); its text encoder
sees text, never the pictures. So the encoders here are: the prompt through the clip, plus the picture(s) VAE-encoded and
attached as `reference_latents`, each with its own strength:

    image_N_strength   1.0 = exactly what the stock ReferenceLatent gives (nothing is changed)
                       0.0 = that reference carries no picture content, above 1 = emphasised
                       (the reference latent is scaled; the same technique community FLUX.2 reference nodes use)

The sampling / crop / paste-back path is shared with Invoke Inpainting (MOSS): MOSS-Image uses the same FLUX.2 VAE (128 latent
channels, 16 px grid), so it is the same geometry."""
import numpy as np
import torch

from .masks import FILL_MODES, append_reference, blank_reference, has_reference_latents, scale_reference
from .nodes import InvokeInpaintingMOSS, _mask_np, zero_conditioning

STRENGTH = {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}


def hidden_reference(image, mode, expand, box_mask=None, inpaint_mask=None):
    """The picture with the area that is about to be repainted hidden (grey / blurred), so the reference does not show the
    thing that is supposed to change. 'keep' or no mask = the picture as it is."""
    img = image[:1, :, :, :3]
    if mode == "keep":
        return img
    _, H, W, _ = img.shape
    box, inp = _mask_np(box_mask, H, W), _mask_np(inpaint_mask, H, W)
    area = None
    if inp is not None and float(inp.max()) > 0.001:
        area = inp * (box > 0.5) if (box is not None and float(box.max()) > 0.5) else inp
    elif box is not None and float(box.max()) > 0.001:
        area = box
    if area is None:
        return img
    arr = img[0].detach().float().cpu().numpy()
    return torch.from_numpy(np.ascontiguousarray(blank_reference(arr, area, mode, int(expand)))).to(image.device, image.dtype)[None]


def encode_flux2(clip, vae, prompt, images, strengths):
    """Prompt -> conditioning, with every given picture attached as a reference latent scaled by its strength. The first picture
    is the one being edited. Strengths are also stored (`invoke_ref_strengths`) so Invoke Inpainting can re-apply them when it
    swaps the reference for the crop it works on."""
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt or ""))
    refs, used = [], []
    for img, s in zip(images, strengths):
        if img is None:
            continue
        lat = vae.encode(img[:1, :, :, :3])
        refs.append(lat if float(s) == 1.0 else lat * float(s))
        used.append(float(s))
    if not refs:
        return cond
    return [[t[0], {**t[1], "reference_latents": refs, "invoke_ref_strengths": used}] for t in cond]


class TextEncodeFlux2Edit:
    """Text Encode for FLUX.2 edit: prompt + up to two pictures as reference latents, each with a strength."""
    CATEGORY = "conditioning/flux2_image"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING",)
    DESCRIPTION = ("FLUX.2 edit encoder: the prompt plus image_0 (the picture being edited) and an optional image_1 (a second reference, "
                   "e.g. a garment or a face) as reference latents, each with its own strength. Use it for the positive and again "
                   "(empty prompt) for the negative. All strengths at 1.0 behave exactly like CLIPTextEncode + ReferenceLatent.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "vae": ("VAE",),
                "image_0": ("IMAGE", {"tooltip": "The picture being edited, used at the size it arrives (scale it before connecting)"}),
                "image_0_strength": ("FLOAT", {**STRENGTH, "tooltip":
                    "How strongly the model follows image_0. 1 = unchanged (stock ReferenceLatent), lower = looser, the edit can change more, 0 = ignored, above 1 = followed harder"}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Optional second reference (garment, face, object...)"}),
                "image_1_strength": ("FLOAT", {**STRENGTH, "tooltip": "How strongly the model follows image_1. 1 = unchanged"}),
            },
        }

    def encode(self, clip, prompt, vae, image_0, image_0_strength=1.0, image_1=None, image_1_strength=1.0):
        return (encode_flux2(clip, vae, prompt, [image_0, image_1], [image_0_strength, image_1_strength]),)


class TextEncodeFlux2Inpaint:
    """TextEncodeFlux2Edit, plus: the part of image_0 that is about to be repainted is hidden from the reference."""
    CATEGORY = "conditioning/flux2_image"
    FUNCTION = "encode"
    RETURN_TYPES = ("CONDITIONING",)
    DESCRIPTION = ("FLUX.2 edit encoder with the edit area hidden from the reference picture (image_0), so the model is not shown the "
                   "old content as the thing to copy and the edit can land, also below strength 1. Plus per-image reference strength. "
                   "Use the same box_mask / inpaint_mask as Invoke Inpainting (FLUX2).")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "vae": ("VAE",),
                "image_0": ("IMAGE", {"tooltip": "The picture being edited (already at the size you want to work at)"}),
                "hide_edit_area": (FILL_MODES, {"default": "grey", "tooltip":
                    "grey: replace the edit area in the reference with flat grey. blurred: a very blurry copy (keeps rough colours/layout). keep: do not hide"}),
                "expand": ("INT", {"default": 16, "min": 0, "max": 256, "step": 8, "tooltip":
                    "Grow the hidden area (and feather its edge) by this many px, so the edit has room to land"}),
                "image_0_strength": ("FLOAT", {**STRENGTH, "tooltip": "How strongly the model follows image_0 (what is left visible of it). 1 = unchanged"}),
            },
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Optional second reference (garment, face, object...). Never hidden"}),
                "image_1_strength": ("FLOAT", {**STRENGTH, "tooltip": "How strongly the model follows image_1. 1 = unchanged"}),
                "box_mask": ("MASK", {"tooltip": "Same as Invoke Inpainting's box_mask. Empty = whole image"}),
                "inpaint_mask": ("MASK", {"tooltip": "Same as Invoke Inpainting's inpaint_mask. Empty = the whole box"}),
            },
        }

    def encode(self, clip, prompt, vae, image_0, hide_edit_area="grey", expand=16, image_0_strength=1.0, image_1=None,
               image_1_strength=1.0, box_mask=None, inpaint_mask=None):
        ref = hidden_reference(image_0, hide_edit_area, expand, box_mask, inpaint_mask)
        return (encode_flux2(clip, vae, prompt, [ref, image_1], [image_0_strength, image_1_strength]),)



class InvokeInpaintingFLUX2(InvokeInpaintingMOSS):
    """Invoke Inpainting for FLUX.2: the same crop / repaint / paste-back as the MOSS node, with FLUX.2's own encoding in per-box
    mode (text through the clip, the picture as a reference latent) and no MOSS-only settings."""
    DESCRIPTION = (
        "InvokeAI-style inpainting for FLUX.2. box_mask = where the edit may happen (crop + paste-back); inpaint_mask = what to repaint "
        "inside the box. Connect positive/negative from the FLUX.2 edit encoders (or stock ReferenceLatent), or clip + passes for "
        "per-box mode. image_0_strength / image_1_strength scale how strongly the references are followed."
    )

    @classmethod
    def INPUT_TYPES(cls):
        base = InvokeInpaintingMOSS.INPUT_TYPES()
        req = dict(base["required"])
        opt = {k: v for k, v in base["optional"].items() if k not in ("edit_format", "prompt_style")}  # MOSS-only
        req["steps"] = ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "FLUX.2 Klein distilled: 4. Base / dev: 20-30"})
        req["cfg"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip":
                                "FLUX.2 Klein distilled: 1 (the negative is not used). Base: about 3-5"})
        opt["positive"] = ("CONDITIONING", {"tooltip": "From Text Encode FLUX.2 Edit (or ReferenceLatent). Not needed in per-box mode (clip + passes)"})
        opt["prompt"] = ("STRING", {"default": "", "multiline": True, "tooltip":
            "Per-box mode: the instruction for the whole picture (connect the bbox node's 'prompt' output, or type it); the scene text for every box. "
            "With no boxes, this is the prompt for one whole-picture pass"})
        opt["image_1"] = ("IMAGE", {"tooltip": "Optional second reference (garment, face, object...). Per-box mode, or added to the connected positive/negative"})
        opt["image_0_strength"] = ("FLOAT", {**STRENGTH, "tooltip":
            "How strongly the model follows the picture being edited. 1 = unchanged. With connected positive/negative, applied to their first reference "
            "(not used on a crop: crop_to_box and enlarged boxes inpaint without the picture being edited as a reference)"})
        opt["image_1_strength"] = ("FLOAT", {**STRENGTH, "tooltip": "How strongly the model follows image_1. 1 = unchanged"})
        return {"required": req, "optional": opt}

    def run(self, image_1=None, image_0_strength=1.0, image_1_strength=1.0, **kw):
        self._f2 = {"image_1": image_1, "s0": float(image_0_strength), "s1": float(image_1_strength)}
        try:
            clip, passes, prompt = kw.get("clip"), kw.get("passes"), (kw.get("prompt") or "")
            per_box = clip is not None and bool(passes or prompt.strip())
            pos, neg, vae = kw.get("positive"), kw.get("negative"), kw.get("vae")
            if not per_box and pos is not None and neg is not None:
                extra = None
                if image_1 is not None and has_reference_latents(pos):
                    lat = vae.encode(image_1[:1, :, :, :3])
                    extra = lat if float(image_1_strength) == 1.0 else lat * float(image_1_strength)
                kw["positive"], kw["negative"] = (self._prepare_plain(c, extra) for c in (pos, neg))
            return super().run(**kw)
        finally:
            self._f2 = {}

    def _prepare_plain(self, cond, extra_latent):
        c = scale_reference(cond, 0, self._f2["s0"]) if has_reference_latents(cond) else cond
        if extra_latent is not None:
            c = append_reference(c, extra_latent, self._f2["s1"])
        return c

    def _encode_pair(self, enc, clip, vae, edit_format, image, hide_edit_area, expand, box_t, inp_t, prompt, negative_prompt, hide=None, reference=True):
        o = getattr(self, "_f2", None) or {"image_1": None, "s0": 1.0, "s1": 1.0}
        img = image
        if hide is not None and float(np.max(hide)) > 0.001:  # items the edit removes: blurred out of the reference only
            arr = image[0].detach().float().cpu().numpy()
            img = torch.from_numpy(np.ascontiguousarray(blank_reference(arr, hide, "blurred", int(expand)))).to(image.device, image.dtype)[None]
        ref = hidden_reference(img, hide_edit_area, expand, box_t, inp_t) if reference else None  # a crop keeps only image_1
        pos = encode_flux2(clip, vae, prompt, [ref, o["image_1"]], [o["s0"], o["s1"]])
        neg = encode_flux2(clip, vae, negative_prompt or "", [ref, o["image_1"]], [o["s0"], o["s1"]])
        if not (negative_prompt or "").strip():
            neg = zero_conditioning(neg)
        return pos, neg


NODE_CLASS_MAPPINGS = {
    "InvokeInpaintingFLUX2": InvokeInpaintingFLUX2,
    "TextEncodeFlux2Edit": TextEncodeFlux2Edit,
    "TextEncodeFlux2Inpaint": TextEncodeFlux2Inpaint,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InvokeInpaintingFLUX2": "Invoke Inpainting (FLUX2)",
    "TextEncodeFlux2Edit": "Text Encode FLUX.2 Edit (reference strength)",
    "TextEncodeFlux2Inpaint": "Text Encode FLUX.2 Edit (masked reference + strength)",
}
