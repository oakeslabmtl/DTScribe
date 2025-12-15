"""
Abstractions and interfaces for SOLID principle compliance.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel
from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for extraction operations."""
    query: str
    description: str
    k: int = 6

@dataclass
class ExtractionResult:
    """Result of an extraction operation."""
    characteristics: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class IDocumentRetriever(ABC):
    """Interface for document retrieval operations."""
    
    @abstractmethod
    def retrieve_documents(self, query: str, k: int = 5, **kwargs) -> List[Any]:
        """Retrieve relevant documents for a query."""
        pass


class ICharacteristicsExtractor(ABC):
    """Interface for characteristics extraction."""
    
    @abstractmethod
    def extract(self, description: str, documents: List[Any], schema: Type[BaseModel], judge_results: List[dict[str, Any]]) -> tuple[BaseModel, dict]:
        """Extract characteristics from documents."""
        pass


class IBlockProcessor(ABC):
    """Interface for processing extraction blocks."""
    
    @abstractmethod
    def get_config(self) -> ExtractionConfig:
        """Get the configuration for this block."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Type[BaseModel]:
        """Get the Pydantic schema for this block."""
        pass
    
    @abstractmethod
    def process(self, retriever: IDocumentRetriever, extractor: ICharacteristicsExtractor) -> ExtractionResult:
        """Process the block and return results."""
        pass


class IIngestor(ABC):
    @abstractmethod
    def ingest(self, path: str, chunk_size: int, overlap: int) -> tuple:
        """Return (vectordb, metadata_dict) for the given path."""
        pass

class IPipelineInitializer(ABC):
    @abstractmethod
    def initialize(self, input_path: str, config: ExtractionConfig, model_name: str, embedding_model: str) -> Dict[str, Any]:
        """Initialize the pipeline with a file or directory path."""
        pass

# class IPipelineInitializer(ABC):
#     """Interface for pipeline initialization."""
    
#     @abstractmethod
#     def initialize(self, pdf_path: str) -> Dict[str, Any]:
#         """Initialize the pipeline with a PDF."""
#         pass


class IOMLGenerator(ABC):
    """Interface for OML generation."""
    
    @abstractmethod
    def generate(self, characteristics: Dict[str, Any], vocab_files: Dict[str, str], max_retries: int = 3):
        """Generate OML from characteristics."""
        pass


class IQualityAnalyzer(ABC):
    """Interface for quality analysis."""
    
    @abstractmethod
    def analyze_characteristics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze extraction quality."""
        pass


class IStateManager(ABC):
    """Interface for state management."""
    
    @abstractmethod
    def get_state(self, key: str) -> Any:
        """Get state value."""
        pass
    
    @abstractmethod
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update state with new values."""
        pass
    
    @abstractmethod
    def merge_characteristics(self, new_characteristics: Dict[str, Any]) -> None:
        """Merge new characteristics with existing ones."""
        pass
