import os
import torch
import requests
import numpy as np
from PIL import Image
import gradio as gr
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig

from image_utils import load_image

if __name__ == "__main__":
    # config
    max_length = 768
    image_size = [720, 960]
    MP = "./donut-save-hf/epoch_0_ned_0.22559996275727548"

    # processor
    processor = DonutProcessor.from_pretrained(MP)
    processor.image_processor.size = image_size[::-1]
    processor.image_processor.do_align_long_axis = False

    # model
    model = VisionEncoderDecoderModel.from_pretrained(MP)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # prompt
    token_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(token_prompt,
                                            add_special_tokens=False,
                                            return_tensors="pt").input_ids.to(device)
    print(f"decoder_input_ids:{decoder_input_ids}")


    def predict(input_image):
        pixel_values = processor.image_processor(input_image, return_tensors="pt").pixel_values.to(device)
        outputs = model.generate(pixel_values=pixel_values,
                                 decoder_input_ids=decoder_input_ids,
                                 max_length=model.decoder.config.max_position_embeddings,
                                 early_stopping=True,
                                 pad_token_id=processor.tokenizer.pad_token_id,
                                 eos_token_id=processor.tokenizer.eos_token_id,
                                 use_cache=True,
                                 num_beams=1,
                                 bad_words_ids=[[processor.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True
                                 )
        prediction = processor.batch_decode(outputs.sequences)[0]
        return str(processor.token2json(prediction))


    # test
    image_path = "./misc/000.jpg"
    image_data, _ = load_image(image_path)

    output = predict(image_data)
    print(output)

    run_gradio = True
    if run_gradio:
        demo = gr.Interface(fn=predict,
                            inputs=gr.inputs.Image(type="pil"),
                            outputs=gr.outputs.Label(num_top_classes=1),
                            examples=[[image_path]]
                            )
        demo.launch()