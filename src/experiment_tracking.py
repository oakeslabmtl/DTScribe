"""
Results saving and evaluation system for research purposes.
Separates characteristics extraction from OML generation with comprehensive metadata.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import hashlib


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_name: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    temperature: float
    top_p: float
    top_k: int
    # repeat_penalty: float
    max_pages: Optional[int] = None
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class CharacteristicsExtractionResult:
    """Results specifically for characteristics extraction task."""
    experiment_id: str
    timestamp: datetime
    pdf_path: str
    config: ExperimentConfig
    
    # Extraction results
    extracted_characteristics: Dict[str, Any]
    extraction_metadata: Dict[str, Any]
    
    # Quality metrics
    total_characteristics: int
    extracted_count: int
    not_found_count: int
    extraction_rate: float
    average_description_length: float
    
    # Performance metrics
    total_docs_retrieved: int
    total_chunks: int
    processing_time_seconds: float
    
    # Block-specific metrics
    block_processing_times: Dict[str, float]
    block_success_rates: Dict[str, bool]
    
    # Error information
    errors: List[str]
    warnings: List[str]


@dataclass
class OMLGenerationResult:
    """Results specifically for OML generation task."""
    experiment_id: str
    timestamp: datetime
    characteristics_experiment_id: str  # Link to source characteristics
    
    # OML generation results
    generated_oml: str
    oml_metadata: Dict[str, Any]
    
    # Quality metrics
    oml_syntax_valid: bool
    oml_completeness_score: float
    oml_line_count: int
    oml_instance_count: int
    
    # Performance metrics
    generation_time_seconds: float
    
    # Error information
    errors: List[str]
    warnings: List[str]


class ResultsSaver:
    """Handles saving and organizing experimental results."""
    
    def __init__(self, base_output_dir: str = "experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.characteristics_dir = self.base_output_dir / "characteristics_extraction"
        self.oml_dir = self.base_output_dir / "oml_generation"
        self.logs_dir = self.base_output_dir / "logs"
        self.analysis_dir = self.base_output_dir / "analysis"
        
        for dir_path in [self.characteristics_dir, self.oml_dir, self.logs_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def generate_experiment_id(self, config: ExperimentConfig, pdf_path: str) -> str:
        """Generate a unique experiment ID based on config and inputs."""
        config_str = json.dumps(asdict(config), sort_keys=True)
        input_str = f"{pdf_path}_{config_str}"
        return hashlib.md5(input_str.encode()).hexdigest()[:12]
    
    def save_characteristics_results(self, result: CharacteristicsExtractionResult) -> Path:
        """Save characteristics extraction results."""
        timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{result.experiment_id}_characteristics.json"
        filepath = self.characteristics_dir / filename
        
        # Convert result to dict for JSON serialization
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Also save a CSV summary for easy analysis
        self._update_characteristics_summary(result)
        
        return filepath
    
    def save_oml_results(self, result: OMLGenerationResult) -> Path:
        """Save OML generation results."""
        timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp_str}_{result.experiment_id}_oml.json"
        filepath = self.oml_dir / filename
        
        # Convert result to dict for JSON serialization
        result_dict = asdict(result)
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        
        # Save the actual OML file separately
        oml_filename = f"{timestamp_str}_{result.experiment_id}.oml"
        oml_filepath = self.oml_dir / oml_filename
        with open(oml_filepath, 'w', encoding='utf-8') as f:
            f.write(result.generated_oml)
        
        # Also save a CSV summary for easy analysis
        self._update_oml_summary(result)
        
        return filepath
    
    def _update_characteristics_summary(self, result: CharacteristicsExtractionResult):
        """Update the characteristics extraction summary CSV."""
        summary_file = self.analysis_dir / "characteristics_summary.csv"
        
        summary_data = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp.isoformat(),
            'pdf_path': result.pdf_path,
            'model_name': result.config.model_name,
            'embedding_model': result.config.embedding_model,
            'chunk_size': result.config.chunk_size,
            'chunk_overlap': result.config.chunk_overlap,
            # 'retrieval_k' removed
            'temperature': result.config.temperature,
            'extraction_rate': result.extraction_rate,
            'extracted_count': result.extracted_count,
            'total_characteristics': result.total_characteristics,
            'average_description_length': result.average_description_length,
            'total_docs_retrieved': result.total_docs_retrieved,
            'processing_time_seconds': result.processing_time_seconds,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings)
        }
        
        # Append to CSV
        df = pd.DataFrame([summary_data])
        if summary_file.exists():
            df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_file, index=False)
    
    def _update_oml_summary(self, result: OMLGenerationResult):
        """Update the OML generation summary CSV."""
        summary_file = self.analysis_dir / "oml_summary.csv"
        
        summary_data = {
            'experiment_id': result.experiment_id,
            'timestamp': result.timestamp.isoformat(),
            'characteristics_experiment_id': result.characteristics_experiment_id,
            'oml_syntax_valid': result.oml_syntax_valid,
            'oml_completeness_score': result.oml_completeness_score,
            'oml_line_count': result.oml_line_count,
            'oml_instance_count': result.oml_instance_count,
            'generation_time_seconds': result.generation_time_seconds,
            'error_count': len(result.errors),
            'warning_count': len(result.warnings)
        }
        
        # Append to CSV
        df = pd.DataFrame([summary_data])
        if summary_file.exists():
            df.to_csv(summary_file, mode='a', header=False, index=False)
        else:
            df.to_csv(summary_file, index=False)
    
    def load_characteristics_results(self, experiment_id: str) -> Optional[CharacteristicsExtractionResult]:
        """Load characteristics results by experiment ID."""
        for file_path in self.characteristics_dir.glob(f"*{experiment_id}_characteristics.json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                return CharacteristicsExtractionResult(**data)
        return None
    
    def get_characteristics_summary(self) -> pd.DataFrame:
        """Get characteristics extraction summary for analysis."""
        summary_file = self.analysis_dir / "characteristics_summary.csv"
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()
    
    def get_oml_summary(self) -> pd.DataFrame:
        """Get OML generation summary for analysis."""
        summary_file = self.analysis_dir / "oml_summary.csv"
        if summary_file.exists():
            return pd.read_csv(summary_file)
        return pd.DataFrame()


class ExperimentTracker:
    """Tracks experiments and their hyperparameters for research purposes."""
    
    def __init__(self, results_saver: ResultsSaver):
        self.results_saver = results_saver
        self.current_config: Optional[ExperimentConfig] = None
    
    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment with given configuration."""
        self.current_config = config
        experiment_id = self.results_saver.generate_experiment_id(config, "")
        
        # Log experiment start
        log_file = self.results_saver.logs_dir / f"{experiment_id}_experiment.log"
        with open(log_file, 'w') as f:
            f.write(f"Experiment {experiment_id} started at {datetime.now().isoformat()}\n")
            f.write(f"Configuration: {json.dumps(asdict(config), indent=2)}\n")
        
        return experiment_id
    
    def create_characteristics_result(self, 
                                    experiment_id: str,
                                    pdf_path: str,
                                    extracted_characteristics: Dict[str, Any],
                                    extraction_metadata: Dict[str, Any],
                                    quality_metrics: Dict[str, Any],
                                    processing_time: float,
                                    block_metrics: Dict[str, Any],
                                    errors: List[str] = None,
                                    warnings: List[str] = None) -> CharacteristicsExtractionResult:
        """Create a characteristics extraction result."""
        
        return CharacteristicsExtractionResult(
            experiment_id=experiment_id,
            timestamp=datetime.now(),
            pdf_path=pdf_path,
            config=self.current_config,
            extracted_characteristics=extracted_characteristics,
            extraction_metadata=extraction_metadata,
            total_characteristics=quality_metrics.get('total_characteristics', 0),
            extracted_count=quality_metrics.get('extracted_count', 0),
            not_found_count=quality_metrics.get('not_found_count', 0),
            extraction_rate=quality_metrics.get('extraction_rate', 0.0),
            average_description_length=quality_metrics.get('average_description_length', 0.0),
            total_docs_retrieved=quality_metrics.get('total_docs_retrieved', 0),
            total_chunks=quality_metrics.get('total_chunks', 0),
            processing_time_seconds=processing_time,
            block_processing_times=block_metrics.get('processing_times', {}),
            block_success_rates=block_metrics.get('success_rates', {}),
            errors=errors or [],
            warnings=warnings or []
        )