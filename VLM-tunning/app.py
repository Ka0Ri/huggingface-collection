import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
from datetime import datetime
import numpy as np
import os


DESCRIPTION = """
# Qwen2-VL-7B-trl-sft-ChartQA Demo
This is a demo Space for a fine-tuned version of [Qwen2-VL-7B](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) trained using [ChatQA dataset](https://huggingface.co/datasets/HuggingFaceM4/ChartQA).
"""

model_id = "Qwen/Qwen2-VL-7B-Instruct" 
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# adapter_path = "sergiopaniego/qwen2-7b-instruct-trl-sft-ChartQA"
# model.load_adapter(adapter_path)
processor = Qwen2VLProcessor.from_pretrained(model_id)

def array_to_image_path(image_array):
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    # Convert numpy array to PIL Image
    img = Image.fromarray(np.uint8(image_array))
    
    # Generate a unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    
    # Save the image
    img.save(filename)
    
    # Get the full path of the saved image
    full_path = os.path.abspath(filename)
    
    return full_path


system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

def run_example(image, text_input=None):
    image_path = array_to_image_path(image)
    image = Image.fromarray(image).convert("RGB")
    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": system_message}],
    },
    {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text", 
                    "text": text_input
                },
            ],
        }
    ]
    
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    return output_text[0]

css = """
  #output {
    height: 500px; 
    overflow: auto; 
    border: 1px solid #ccc; 
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tab(label="Qwen2-VL-7B-trl-sft-ChartQA Input"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(label="Input Picture")
                text_input = gr.Textbox(label="Question")
                submit_btn = gr.Button(value="Submit")
            with gr.Column():
                output_text = gr.Textbox(label="Output Text")

        submit_btn.click(run_example, [input_img, text_input], [output_text])

demo.queue(api_open=False)
demo.launch(debug=True, share=True)