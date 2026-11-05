"""FLUX.2 nodes: wiring and logic with ComfyUI/torch stubbed out (run: python -m unittest discover -s tests)."""
import importlib
import os
import sys
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    stubs = {}
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except Exception:
            t = types.ModuleType("torch"); t.zeros_like = np.zeros_like; t.from_numpy = lambda a: a; t.Tensor = np.ndarray
            nn = types.ModuleType("torch.nn"); f = types.ModuleType("torch.nn.functional"); nn.functional = f; t.nn = nn
            stubs.update({"torch": t, "torch.nn": nn, "torch.nn.functional": f})
    stubs.setdefault("nodes", types.ModuleType("nodes"))
    sys.modules.update({k: v for k, v in stubs.items() if k not in sys.modules or k == "nodes"})
    pkg = types.ModuleType("invoke_inpainting_t"); pkg.__path__ = [ROOT]; sys.modules["invoke_inpainting_t"] = pkg
    return importlib.import_module("invoke_inpainting_t.flux2"), importlib.import_module("invoke_inpainting_t.masks")


f2, masks = _load()


class FakeClip:
    def tokenize(self, text): return text
    def encode_from_tokens_scheduled(self, tokens): return [[np.ones((1, 4)), {"pooled_output": np.ones((1, 2)), "text": tokens}]]


class FakeVae:
    def encode(self, img): return np.asarray(img, dtype=np.float32)[:, ::16, ::16, :1]  # any deterministic tensor


class InputTypes(unittest.TestCase):
    def test_flux2_node_drops_moss_only_and_adds_strengths(self):
        it = f2.InvokeInpaintingFLUX2.INPUT_TYPES()
        opt = list(it["optional"])
        self.assertNotIn("edit_format", opt); self.assertNotIn("prompt_style", opt)
        for k in ("image_1", "image_0_strength", "image_1_strength", "clip", "passes", "positive", "negative"):
            self.assertIn(k, opt)
        base = f2.InvokeInpaintingMOSS.INPUT_TYPES()
        self.assertEqual(list(it["required"]), list(base["required"]))  # same widget order up front (no shifting)
        self.assertEqual(it["required"]["steps"][1]["default"], 20)

    def test_run_signature_accepts_every_input(self):
        import inspect
        it = f2.InvokeInpaintingFLUX2.INPUT_TYPES()
        names = set(it["required"]) | set(it["optional"])
        accepted = set(inspect.signature(f2.InvokeInpaintingMOSS.run).parameters) | {"image_1", "image_0_strength", "image_1_strength"}
        self.assertEqual(names - accepted, set())


class Encode(unittest.TestCase):
    def test_strength_1_is_a_plain_reference(self):
        img = np.random.rand(1, 64, 64, 3).astype(np.float32)
        c = f2.encode_flux2(FakeClip(), FakeVae(), "red dress", [img, None], [1.0, 1.0])
        d = c[0][1]
        self.assertEqual(len(d["reference_latents"]), 1)
        np.testing.assert_allclose(d["reference_latents"][0], FakeVae().encode(img))  # untouched
        self.assertEqual(d["invoke_ref_strengths"], [1.0])

    def test_two_images_scaled_independently(self):
        a, b = np.random.rand(1, 64, 64, 3).astype(np.float32), np.random.rand(1, 32, 32, 3).astype(np.float32)
        d = f2.encode_flux2(FakeClip(), FakeVae(), "x", [a, b], [0.5, 2.0])[0][1]
        np.testing.assert_allclose(d["reference_latents"][0], FakeVae().encode(a) * 0.5, rtol=1e-6)
        np.testing.assert_allclose(d["reference_latents"][1], FakeVae().encode(b) * 2.0, rtol=1e-6)
        self.assertEqual(d["invoke_ref_strengths"], [0.5, 2.0])

    def test_no_images_is_just_text(self):
        c = f2.encode_flux2(FakeClip(), FakeVae(), "x", [None, None], [1.0, 1.0])
        self.assertNotIn("reference_latents", c[0][1])

    def test_zero_conditioning_keeps_references(self):
        img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        t = sys.modules["torch"]  # real torch or the stub
        c = f2.encode_flux2(FakeClip(), FakeVae(), "x", [img], [1.0])
        z = f2.zero_conditioning([[t.from_numpy(c[0][0]), {**c[0][1], "pooled_output": t.from_numpy(c[0][1]["pooled_output"])}]])
        self.assertEqual(float(z[0][0].sum()), 0.0)
        self.assertEqual(len(z[0][1]["reference_latents"]), 1)

    def test_encode_nodes_run(self):
        img = np.random.rand(1, 64, 64, 3).astype(np.float32)
        out = f2.TextEncodeFlux2Edit().encode(FakeClip(), "x", FakeVae(), img, 0.7, img, 1.3)[0]
        self.assertEqual(out[0][1]["invoke_ref_strengths"], [0.7, 1.3])
        out = f2.TextEncodeFlux2Inpaint().encode(FakeClip(), "x", FakeVae(), img, "keep", 16, 0.7)[0]
        self.assertEqual(out[0][1]["invoke_ref_strengths"], [0.7])


class ScaleAndAppend(unittest.TestCase):
    def test_scale_and_append_on_stock_reference_latent(self):
        stock = [["e", {"reference_latents": [np.ones(3)]}]]
        scaled = masks.scale_reference(stock, 0, 0.5)
        self.assertEqual(scaled[0][1]["invoke_ref_strengths"], [0.5])
        np.testing.assert_allclose(scaled[0][1]["reference_latents"][0], np.full(3, 0.5))
        np.testing.assert_allclose(stock[0][1]["reference_latents"][0], np.ones(3))  # original untouched
        both = masks.append_reference(scaled, np.full(3, 9.0), 1.5)[0][1]
        self.assertEqual(len(both["reference_latents"]), 2); self.assertEqual(both["invoke_ref_strengths"], [0.5, 1.5])
        self.assertIs(masks.scale_reference(stock, 0, 1.0)[0], stock[0])  # 1.0 = exact pass-through


class PlainModePrep(unittest.TestCase):
    def test_prepare_plain(self):
        n = f2.InvokeInpaintingFLUX2()
        n._f2 = {"image_1": None, "s0": 0.5, "s1": 2.0}
        stock = [["e", {"reference_latents": [np.ones(3)]}]]
        out = n._prepare_plain(stock, np.full(3, 4.0))[0][1]
        self.assertEqual(len(out["reference_latents"]), 2)
        np.testing.assert_allclose(out["reference_latents"][0], np.full(3, 0.5))
        no_ref = [["e", {}]]
        self.assertEqual(n._prepare_plain(no_ref, None), no_ref)  # t2i conditioning is left alone


if __name__ == "__main__":
    unittest.main()
