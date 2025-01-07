from smolagents import CodeAgent
from google.cloud import storage, documentai_v1 as documentai
from typing import Dict, Any, List, Optional
import os
import json
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModel
import torch
from trademarkpredictor.config import get_config
from trademarkpredictor.trademark_analyzer import TrademarkAnalyzer
from trademarkpredictor.trademark_schema_extractor import TrademarkVectorSystem
from trademarkpredictor.document_processor import TrademarkDocumentProcessor


class TrademarkCaseAgent:
    """
    An agent for processing trademark dispute cases using the smolagents framework.
    Integrates with existing configuration and handles structured metadata.
    """

    def __init__(self):
        # Load configuration
        self.config = get_config()

        # Initialize cloud clients
        self.storage_client = storage.Client(project=self.config["PROJECT_ID"])
        self.documentai_client = documentai.DocumentProcessorServiceClient()

        # Initialize Pinecone with configuration
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        if self.config["INDEX_NAME"] not in self.pc.list_indexes().names():
            self._create_pinecone_index()
        self.index = self.pc.Index(self.config["INDEX_NAME"])

        # Initialize LEGAL-BERT for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.model = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

        # Initialize smolagents CodeAgent
        self.agent = CodeAgent(
            tools=[
                self.validate_case_data,
                self.process_document,
                self.generate_embedding,
                self.store_in_pinecone,
            ]
        )

    def _create_pinecone_index(self) -> None:
        """Creates the Pinecone index if it doesn't exist."""
        self.pc.create_index(
            name=self.config["INDEX_NAME"],
            dimension=self.config["DIMENSION"],
            metric="cosine",
            spec={
                "cloud": self.config["CLOUD"],
                "region": self.config["REGION"],
            },
        )

    def validate_case_data(self, case_data: Dict[str, Any]) -> bool:
        """Validates case data against schema."""
        try:
            with open(self.config["SCHEMA_PATH"], "r") as schema_file:
                schema = json.load(schema_file)
            required_keys = [
                "case_metadata",
                "party_info",
                "commercial_context",
                "similarity_assessment",
                "outcome",
            ]
            return all(key in case_data for key in required_keys)
        except Exception as e:
            print(self.config["ERROR_MESSAGES"]["SCHEMA_READ_ERROR"].format(str(e)))
            return False

    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flattens nested metadata while preserving arrays of primitives and converting other complex types to strings."""
        flat_data: Dict[str, Any] = {}

        def flatten(data: Dict[str, Any], prefix: str = "") -> None:
            for key, value in data.items():
                new_key = f"{prefix}_{key}" if prefix else key

                if isinstance(value, dict):
                    flatten(value, new_key)
                elif isinstance(value, list):
                    # Keep arrays of primitives intact
                    if all(isinstance(x, (str, int, float, bool)) for x in value):
                        flat_data[new_key] = value
                    else:
                        flat_data[new_key] = str(value)
                elif isinstance(value, (str, int, float, bool)):
                    flat_data[new_key] = value
                else:
                    flat_data[new_key] = str(value)

        flatten(metadata)
        return flat_data

    def store_in_pinecone(self, case_id: str, embedding: List[float], metadata: Dict[str, Any]) -> bool:
        """
        Stores vector and metadata in Pinecone with proper metadata handling.
        """
        try:
            flat_metadata = self._flatten_metadata(metadata)
            self.index.upsert(
                vectors=[
                    {
                        "id": case_id,
                        "values": embedding,
                        "metadata": flat_metadata,
                    }
                ]
            )
            return True
        except Exception as e:
            print(f"Pinecone storage error: {e}")
            return False

    def process_document(self, file_path: str) -> Optional[str]:
        """
        Processes a PDF document and returns extracted text using Google Document AI.
        """
        try:
            with open(file_path, "rb") as pdf_file:
                content = pdf_file.read()

            request = documentai.types.ProcessRequest(
                raw_document=documentai.types.RawDocument(content=content)
            )
            response = self.documentai_client.process_document(request=request)
            return response.document.text
        except Exception as e:
            print(f"Document processing error: {e}")
            return None

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generates an embedding from the given text using LEGAL-BERT.
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            print(f"Embedding generation error: {e}")
            return None

    def process_case(self, case_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a case using the smolagents CodeAgent.
        """
        try:
            # First validate the case data
            if not self.validate_case_data(case_data):
                return {
                    "success": False,
                    "error": "Case data validation failed",
                }

            # Generate embedding from the text representation
            text_representation = self._create_text_representation(case_data)
            embedding = self.generate_embedding(text_representation)

            if not embedding:
                return {
                    "success": False,
                    "error": self.config["ERROR_MESSAGES"]["EMBEDDING_FAILED"],
                }

            # Store in Pinecone
            success = self.store_in_pinecone(
                case_id=case_data["case_metadata"]["case_ref"],
                embedding=embedding,
                metadata=case_data,
            )

            return {
                "success": success,
                "case_ref": case_data["case_metadata"]["case_ref"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": self.config["ERROR_MESSAGES"]["UNEXPECTED_ERROR"].format(str(e)),
            }

    def _create_text_representation(self, case_data: Dict[str, Any]) -> str:
        """
        Creates a text representation of the case for embedding generation.
        """
        text_parts = []

        # Add case metadata
        meta = case_data["case_metadata"]
        text_parts.append(f"Case {meta['case_ref']} decided by {meta['officer']} on {meta['dec_date']}")

        # Add party information
        party = case_data["party_info"]
        text_parts.append(
            f"Application for '{party['app_mark']}' by {party['app_name']} "
            f"opposed by {party['opp_name']} based on '{party['opp_mark']}'"
        )

        # Add commercial context
        comm = case_data["commercial_context"]
        text_parts.append(
            f"Application covers {comm['app_spec']} in class {', '.join(map(str, comm['app_class']))}. "
            f"Opposition based on {comm['opp_spec']} in class {', '.join(map(str, comm['opp_class']))}"
        )

        return " ".join(text_parts)
