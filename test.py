import onnxruntime as ort

session = ort.InferenceSession("best.onnx")

print("Inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("\nOutputs:")
for out in session.get_outputs():
    print(out.name, out.shape, out.type)