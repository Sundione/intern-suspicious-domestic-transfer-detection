import time
import logging
from typing import Optional
from google import genai
from google.genai import types
import base64
from adapter.genai.model.exception import (
    InvalidGenAIInput,
    InvalidCacheInput,
    InvalidGenAIConfiguration,
)


class BaseGenAI:

    def __init__(
        self,
        genai_api_key: str,
        genai_model: Optional[str] = "gemini-2.0-flash",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.genai_api_key = genai_api_key
        self.genai_model = genai_model
        print(f"Init Google AI with model : {genai_model}")
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def generate(
        self, instruction=None, prompt=None, image_bytes=None, cache_display_name=None
    ):
        response = None
        try:
            client = genai.Client(
                api_key=self.genai_api_key,
            )
            parts = []
            if prompt:
                parts.append(types.Part.from_text(text=prompt))
            if image_bytes:
                parts.append(
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/png",
                            data=base64.b64decode(image_bytes),
                        )
                    ),
                )
            if not parts:
                raise InvalidGenAIInput(
                    "At least one of 'image_bytes' or 'prompt' must be provided."
                )
            contents = [
                types.Content(
                    role="user",
                    parts=parts,
                ),
            ]
            config = None
            if cache_display_name and (client.caches.list()):
                for cache in client.caches.list():
                    if cache.display_name == cache_display_name:
                        print(
                            f"Cache with display name {cache_display_name} found: {cache.name}"
                        )
                        config = types.GenerateContentConfig(cached_content=cache.name)
                    else:
                        raise InvalidCacheInput(
                            f"Cache with display name '{cache_display_name}' not found."
                        )
            elif cache_display_name and not (client.caches.list()):
                raise InvalidCacheInput("No caches found. Please create a cache first.")
            elif instruction:
                config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    system_instruction=[
                        types.Part.from_text(text=instruction),
                    ],
                )
            if not config:
                raise InvalidGenAIConfiguration(
                    "Either 'cache_display_name' or 'instruction' must be provided."
                )
            start_time = time.perf_counter()

            response = client.models.generate_content(
                model=self.genai_model, contents=contents, config=config
            )

            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"Google Gen AI : \t\tTook {total_time:.4f} seconds")
            return response
        except Exception as e:
            print(f"Google AI Error : {e}")
            raise
