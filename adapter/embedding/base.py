import time
import logging
from typing import Optional, List
from google import genai
from google.genai import types
from google.genai.errors import APIError
from adapter.embedding.model.exception import (
    InvalidEmbeddingInput,
    InvalidEmbeddingClientResponse,
)


class BaseGoogleEmbedding:

    def __init__(
        self,
        embedding_api_key: str,
        embedding_model: Optional[str] = "gemini-embedding-001",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.client = genai.Client(
            api_key=self.embedding_api_key,
        )
        print(f"Init Google Embedding with model : {embedding_model}")
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def embedding(
        self,
        input_texts: List[str],
        task_type: str = "CLASSIFICATION",
        output_dimension: int = 128,
    ):
        if not isinstance(input_texts, list) or len(input_texts) == 0:
            raise InvalidEmbeddingInput(
                "Input text must be list and cannot be empty list."
            )
        if not all(isinstance(t, str) for t in input_texts):
            raise InvalidEmbeddingInput("All items in the input list must be strings.")
        try:
            start_time = time.perf_counter()

            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=input_texts,
                config=types.EmbedContentConfig(
                    task_type=task_type, outputDimensionality=output_dimension
                ),
            )

            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"Google Embedding : \t\tTook {total_time:.4f} seconds")

            if not result.embeddings or len(result.embeddings) != len(input_texts):
                raise InvalidEmbeddingClientResponse(
                    "Mismatched length of embeddings received from API."
                )

            embeds = [e.values for e in result.embeddings]
            for vec in embeds:
                if len(vec) != output_dimension:
                    raise InvalidEmbeddingClientResponse(
                        f"Embedding dimension mismatch: expected {output_dimension}, got {len(vec)}"
                    )
            return embeds

        except APIError as e:
            print(f"Google Embedding API Error : {e}")
            raise

        except (InvalidEmbeddingInput, InvalidEmbeddingClientResponse):
            raise

        except Exception as e:
            print(f"Google Embedding Error : {e}")
            raise
