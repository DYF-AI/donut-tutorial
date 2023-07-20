from transformers import VisionEncoderDecoderModel

if __name__ == "__main__":
    MP = "./donut-save-hf/epoch_0_ned_0.22559996275727548"
    model = VisionEncoderDecoderModel.from_pretrained(MP)

    # https://github.com/huggingface/transformers/issues/19983