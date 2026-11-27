"""Export the trained D-FINE model to ONNX, so the robot needs only onnxruntime (MIT) - no PyTorch, no transformers.

  ~/isaacsim/python.sh free_detector/export_onnx.py                          # models/dfine_s_egg_ball
  ~/isaacsim/python.sh free_detector/export_onnx.py --model free_detector/models/<name>
  -> <model folder>/model.onnx + labels.json

ONNX graph: input  "images"  float32 (1, 3, 640, 640), RGB, values 0..1 (image simply resized to 640 x 640)
            output "scores"  float32 (1, 300, classes)  sigmoid probabilities per query and class
                   "boxes"   float32 (1, 300, 4)        cx, cy, w, h normalised 0..1
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import DFineForObjectDetection

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(HERE / "models" / "dfine_s_egg_ball"))
ap.add_argument("--out", default=None, help="default <model>/model.onnx")
args = ap.parse_args()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, images):
        o = self.m(pixel_values=images)
        return o.logits.sigmoid(), o.pred_boxes


model = DFineForObjectDetection.from_pretrained(args.model).eval()
out = Path(args.out) if args.out else Path(args.model) / "model.onnx"
x = torch.rand(1, 3, 640, 640)
with torch.no_grad():
    ref = Wrap(model)(x)
    torch.onnx.export(Wrap(model), (x,), str(out), input_names=["images"], output_names=["scores", "boxes"],
                      opset_version=17, dynamo=False)
labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
(out.parent / "labels.json").write_text(json.dumps(labels))
print(f"exported {out} ({out.stat().st_size / 1e6:.1f} MB), classes {labels}")

try:  # check the ONNX file gives the same numbers as PyTorch
    import onnxruntime as ort
    s, b = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})
    print(f"onnxruntime check: max diff scores {abs(s - ref[0].numpy()).max():.2e}, boxes {abs(b - ref[1].numpy()).max():.2e}")
except ImportError:
    print("(onnxruntime not installed here - checked later by the ROS node)")
