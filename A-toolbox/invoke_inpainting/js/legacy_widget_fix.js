import { app } from "../../scripts/app.js";

// The old "Advanced settings..." button widget was inserted into the middle of these nodes' widget list. That shifted every
// widget after it by one position, so workflows saved with it could load with values in the wrong slots (cfg = None,
// coherence_mode = 1, minimum_denoise = "Gaussian Blur", ...). The button is gone and every setting is always visible.
// This only repairs workflows that were saved while the button existed: if the saved values carry one extra null where
// the button used to be, that slot is dropped before the values are applied. Nothing here touches the widgets themselves.
const BUTTON_SLOT = { InvokeInpaintingMOSS: 3, SceneContextQwenVL: 1 };

app.registerExtension({
  name: "midfr.legacy_widget_fix",
  nodeCreated(node) {
    const slot = BUTTON_SLOT[node.comfyClass || node.type];
    if (slot === undefined || typeof node.configure !== "function") return;
    const configure = node.configure;
    node.configure = function (info, ...rest) {
      try {
        const v = info && info.widgets_values;
        if (Array.isArray(v) && this.widgets && v.length === this.widgets.length + 1 && v[slot] == null) {
          info = { ...info, widgets_values: v.filter((_, i) => i !== slot) };
        }
        if (info && info.properties && "advanced" in info.properties) {
          const { advanced, ...props } = info.properties;
          info = { ...info, properties: props };
        }
      } catch (e) { /* never block loading a workflow */ }
      return configure.call(this, info, ...rest);
    };
  },
});
