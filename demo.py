import os
import numpy as np
import torch
from loguru import logger
# For loading in the tiny-LLaVA-v1-hf model in a transformers pipeline.
import transformers
from transformers import pipeline
from transformers import BitsAndBytesConfig

# For converting input images to PIL images.
from PIL import Image

# For creating the gradio app.
import gradio as gr

# For creating a simple prompt (open to extension) to our model.
from langchain.prompts import PromptTemplate

# Our vector database of choice: Chroma!
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# For loading in our OpenAI API key.
# from google.colab import userdata

from dotenv import load_dotenv


# Required for us to load in our pipeline for TinyLLaVA.
assert transformers.__version__ >= "4.35.3"

model_id = "bczhou/tiny-llava-v1-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

pipe = pipeline(
    "image-to-text",
    model=model_id,
    device_map="auto",
    use_fast=True,
    model_kwargs={"quantization_config": bnb_config}
)
load_dotenv()
# Use OpenAI's embeddings for our Chroma collection.
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY"),# type: ignore
)
collection = Chroma("conversation_memory", embeddings)

# img_data = requests.get("https://imgur.com/Ca6gjuf.png").content
# with open('sample_image.png', 'wb') as handler:
#     handler.write(img_data)

max_new_tokens = 200

# Path for storing images.
IMG_ROOT_PATH = "data/"
os.makedirs(IMG_ROOT_PATH, exist_ok=True)

# Define the function with (message, history) + additional_inputs -> str.
def generate_output(message: str, history: list, img: np.ndarray) -> str:
    """Generates an output given a message and image."""

    # Get detailed description of the image for Chroma.
    query = "Please provide a detailed description of the image."
    img_prompt_template = PromptTemplate.from_template(
        "USER: <image>\n" +
        "{query}" +
        "\n" +
        "ASSISTANT: "
    )
    img_prompt = img_prompt_template.format(query=query)
    logger.info(f' ==== img_prompt ====\n{img_prompt}')
    try:
        outputs = pipe(Image.fromarray(img), prompt=img_prompt, generate_kwargs={"max_new_tokens": max_new_tokens})
        logger.debug(f' ==== img_outputs ====\n{outputs}')
        img_desc = outputs[0]["generated_text"].split("ASSISTANT:")[-1]# type: ignore
        img_desc = img_desc.strip()
    except Exception as e:
        logger.error(e)
        img_desc = ""
    logger.info(f' ==== img_desc ====\n{img_desc}')

    # Visual Question-Answering!
    vqa_prompt_template = PromptTemplate.from_template(
        "Context: {context}\n\n"
        "USER: <image>\n" +
        "{message}" +
        "\n" +
        "ASSISTANT: "
    )
    context = collection.similarity_search(query=message, k=2)
    context = "\n".join([doc.page_content for doc in context])
    logger.info(f' ==== find context from vector storage ====\n{context}')

    # Forward pass through the model with given prompt template.
    vqa_prompt = vqa_prompt_template.format(
                context=context,
                message=message
            )
    logger.info(f' ==== vqa_prompt ====\n{vqa_prompt}')
    try:
        outputs = pipe(
            Image.fromarray(img),
            prompt=vqa_prompt,
            generate_kwargs={"max_new_tokens": max_new_tokens}
        )
        logger.debug(f' ==== vqa_outputs ====\n{outputs}')
        response = outputs[0]["generated_text"].split("ASSISTANT:")[-1]# type: ignore
        response = response.strip()
    except Exception as e:
      logger.error(e)
      response = ""
    logger.info(f' ==== vqa_response ====\n{response}')
    # Add (img_desc, message, response) 3-tuple to Chroma collection.
    text = f"Image Description: {img_desc}\nUSER: {message}\nASSISTANT: {response}\n"
    logger.debug(f' ==== add texts to vector storage ====\n{text}')
    collection.add_texts(texts=[text])

    # Return model output.
    return f"**Image description**\n{img_desc}\n**Response**\n{response}"


# Define the ChatInterface, customize, and launch!
gr.ChatInterface(
    generate_output,
    chatbot=gr.Chatbot(
        label="Chat with me!",
      ),
    textbox=gr.Textbox(
        placeholder="Message ...",
        scale=7,
        info="Input your textual response in the text field and your image below!"
    ),
    additional_inputs="image",
    additional_inputs_accordion=gr.Accordion(
        open=True,
    ),
    title="Language-Image Question Answering with bczhou/TinyLLaVA-v1-hf!",
    description="""
    This simple gradio app internally uses a Large Language-Vision Model (LLVM) and the Chroma vector database for memory.
    Note: this minimal app requires both an image and a text-based query before the chatbot system can respond.
    """,
    submit_btn="Submit",
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch(debug=True, share=False, server_port = 6006)