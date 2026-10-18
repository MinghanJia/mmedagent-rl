from pathlib import Path
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_path = "/data/jiaminghan/models/Qwen2.5-VL-7B-Instruct"
root = Path("/home/jiaminghan/mmedagent-rl")
torch.manual_seed(42)

assert torch.cuda.is_available(), "CUDA 不可用"
print("GPU:", torch.cuda.get_device_name(0), flush=True)

image = Image.new("RGB", (448, 448), "white")
ImageDraw.Draw(image).rectangle((112, 112, 336, 336), fill="red")
image.save(root / "logs/test_image.png")

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
).eval()
print("Model loaded.", flush=True)

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {
            "type": "text",
            "text": "Describe the shape and its color in this image in one sentence.",
        },
    ],
}]

prompt = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
inputs = processor(
    text=[prompt],
    images=[image],
    return_tensors="pt",
).to("cuda:0")

assert "pixel_values" in inputs, "缺少图像输入"
print("Image grid:", inputs["image_grid_thw"].tolist(), flush=True)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
    )

new_tokens = output[:, inputs["input_ids"].shape[1]:]
answer = processor.batch_decode(
    new_tokens, skip_special_tokens=True
)[0].strip()
assert answer, "模型输出为空"

print("Answer:", answer)
print(
    "Peak allocated GPU memory (GiB):",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
)
print("Inference execution: PASS")
