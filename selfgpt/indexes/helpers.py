import base64
import io
import logging
import os
import re
import tempfile

import nltk
import numpy as np
import torch
import whisper
import yt_dlp
from nltk.tokenize import sent_tokenize
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from config.settings.base import MODELS_DIR

logger = logging.getLogger(__name__)


class NLPTools:
    _tokenizer = None
    _model = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            nltk.download("punkt", download_dir=MODELS_DIR, quiet=True)
            cls._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return cls._tokenizer

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        return cls._model


def preprocess_text(text):
    # Remove excessive whitespaces
    text = re.sub(r"\s+", " ", text)
    # Additional preprocessing steps can be added here
    return text.strip()


def bytes_to_string(byte_content):
    try:
        # Assuming the file is encoded in UTF-8
        text = byte_content.decode("utf-8")
    except UnicodeDecodeError as e:
        # Handle the error or try a different encoding
        logging.error(f"Error decoding bytes: {e}")
        text = None  # or handle differently
    return text


def create_offset_sentence_chunks(text, chunk_size=4, offset=2, long_context_size=10):
    sentences = sent_tokenize(text)
    chunks = []

    # Standard and offset chunks
    for i in range(0, len(sentences), chunk_size):
        if i + chunk_size <= len(sentences):
            chunk = " ".join(sentences[i : i + chunk_size])
            chunks.append(preprocess_text(chunk))

    if offset != 0:
        for i in range(offset, len(sentences), chunk_size):
            if i + chunk_size <= len(sentences):
                offset_chunk = " ".join(sentences[i : i + chunk_size])
                chunks.append(preprocess_text(offset_chunk))

    # Additional step for longer context windows
    for i in range(0, len(sentences), long_context_size):
        if i + long_context_size <= len(sentences):
            long_context_chunk = " ".join(sentences[i : i + long_context_size])
            chunks.append(preprocess_text(long_context_chunk))

    # If we are not able to split the chunks, return it in raw form
    if len(sentences) == 1:
        return sentences

    return chunks


def vectorize_chunks(chunks, batch_size=8):
    if not isinstance(chunks, list) or len(chunks) == 0:
        return []

    tokenizer = NLPTools.get_tokenizer()
    model = NLPTools.get_model()

    all_embeddings = []  # Store all embeddings here
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]

        # Tokenize the batch of chunks
        inputs = tokenizer(batch_chunks, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Move inputs to the same device as model
        inputs = {key: val.to(model.device) for key, val in inputs.items()}

        # Get the model's output for the batch
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract embeddings, e.g., by averaging token embeddings for each chunk in the batch
        batch_embeddings = outputs.last_hidden_state.mean(dim=1)

        # Normalize the embeddings to have unit norm
        norms = torch.norm(batch_embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = batch_embeddings / norms

        all_embeddings.append(normalized_embeddings.cpu().numpy())

    # Concatenate all batch embeddings into a single numpy array
    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


def encode_resize_image(image_path, max_size=600):
    # Open the image and resize if necessary
    with Image.open(image_path) as img:
        original_width, original_height = img.size

        # Determine if resizing is needed
        if original_width > max_size or original_height > max_size:
            if original_width > original_height:
                scale_factor = max_size / original_width
                new_width = max_size
                new_height = int(original_height * scale_factor)
            else:
                scale_factor = max_size / original_height
                new_height = max_size
                new_width = int(original_width * scale_factor)

            img = img.resize((new_width, new_height), Image.LANCZOS)

        # Convert the PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # Encode the image bytes to base64
        return base64.b64encode(img_byte_arr).decode("utf-8")


def transcribe_video(youtube_url):
    key = youtube_url.split("v=")[-1]
    logger.info(f"transcribe_video get new URL for processing: {youtube_url}, will use key: {key}")

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": {"default": os.path.join(tmpdir, "%(id)s.%(ext)s")},
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info(youtube_url, download=False)

                video_title = info_dict.get("title", "Unknown Title")
                duration = info_dict.get("duration", 0)

                if duration > 8400:
                    raise ValueError(f"Video is longer than 140 minutes. Title: {video_title}")

                logger.info(f"transcribe_video {key} audio info loaded. Title: {video_title}, length: {duration}")

                try:
                    ydl.download([youtube_url])
                except Exception as download_error:
                    logger.error(f"Error during download: {str(download_error)}")
                    raise Exception(f"Failed to download audio: {str(download_error)}")

                audio_file = os.path.join(tmpdir, f"{key}.mp3")

                if not os.path.exists(audio_file):
                    raise FileNotFoundError(f"Audio file not found: {audio_file}")

                whisper_model = whisper.load_model("base", download_root=MODELS_DIR)
                logger.info(f"transcribe_video ({key}) model loaded")

                transcription = whisper_model.transcribe(audio_file, fp16=False)["text"].strip()
                cleaned_transcription = transcription.encode("utf-8", errors="ignore").decode("utf-8")
                logger.info(f"transcribe_video ({key}) transcription performed")

        except Exception as e:
            logger.error(f"transcribe_video problem with {youtube_url} - {e}")
            raise Exception(f"Problem with video file. Check the link you provided! Error: {str(e)}")

    return cleaned_transcription, video_title
