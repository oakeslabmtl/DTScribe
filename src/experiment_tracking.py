"""
Results saving and evaluation system for research purposes.
Separates characteristics extraction from OML generation with comprehensive metadata.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict, field
import pandas as pd
import hashlib

# Experiment id format: <12-char config hash>_<YYYYMMDDHHMMSS>

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
    max_pages: Optional[int] = None
    max_judge_retries: int = 2
    max_oml_retries: int = 2
    custom_params: Optional[Dict[str, Any]] = None


@dataclass
class CharacteristicsExtractionResult:
    """Results specifically for characteristics extraction task."""
    experiment_id: str
    timestamp: datetime
    input_path: str
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
    total_chunks: int
    processing_time_seconds: float
    
    # Block-specific metrics
    block_processing_times: Dict[str, float]
    block_success_rates: Dict[str, bool]

    # token usage
    total_input_tokens: int
    total_output_tokens: int #Optional[Dict[str, int]] = None

    # llm judge usage
    # judged_characteristics: Dict[str, Any] = field(default_factory=dict)
    block_retries: Dict[str, int] = field(default_factory=dict)
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


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
    oml_valid: bool
    oml_line_count: int
    oml_instance_count: int
    
    # Performance metrics
    generation_time_seconds: float
    oml_max_retries: int
    oml_repetition_count: int
    
    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


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
    
    def generate_experiment_id(self, config: ExperimentConfig, input_path: str) -> str:
        """Generate a readable, unique experiment ID.

        Format: <12hexhash>_<YYYYMMDDHHMMSS>
        - 12hexhash: stable hash of (input_path + config) enabling grouping
        - timestamp: ensures uniqueness per run even if config unchanged
        """
        config_str = json.dumps(asdict(config), sort_keys=True)
        input_str = f"{input_path}_{config_str}"
        base_hash = hashlib.md5(input_str.encode()).hexdigest()[:12]
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{base_hash}_{ts}"
    
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
            'input_path': result.input_path,
            'model_name': result.config.model_name,
            'embedding_model': result.config.embedding_model,
            'chunk_size': result.config.chunk_size,
            'chunk_overlap': result.config.chunk_overlap,
            'temperature': result.config.temperature,
            'extraction_rate': result.extraction_rate,
            'extracted_count': result.extracted_count,
            'total_characteristics': result.total_characteristics,
            'average_description_length': result.average_description_length,
            'processing_time_seconds': result.processing_time_seconds,
            'total_input_tokens': result.total_input_tokens,
            'total_output_tokens': result.total_output_tokens,
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
            'oml_valid': result.oml_valid,
            'oml_line_count': result.oml_line_count,
            'oml_instance_count': result.oml_instance_count,
            'oml_repetition_count': result.oml_repetition_count,
            'oml_max_retries': result.oml_max_retries,
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
        """Load characteristics results by experiment ID.

        Backward compatible:
        - If full new-format ID (hash_timestamp) supplied, exact match.
        - If only the 12-char hash supplied, return the latest run (by filename timestamp prefix).
        - If old style (just 12-char hash) files exist, still matched.
        """
        # Full id if it contains an underscore and length > 13
        if '_' in experiment_id and len(experiment_id) > 13:
            pattern = f"*{experiment_id}_characteristics.json"
            candidates = list(self.characteristics_dir.glob(pattern))
        else:
            # Base hash only: collect all runs whose experiment_id starts with that hash
            # Filenames: <filets>_<experiment_id>_characteristics.json
            # We look for _<hash>_ pattern to reduce false positives.
            candidates = list(self.characteristics_dir.glob(f"*_{experiment_id}_*_characteristics.json"))
            if not candidates:
                # Fallback for legacy files where experiment_id was only the hash
                candidates = list(self.characteristics_dir.glob(f"*{experiment_id}_characteristics.json"))

        if not candidates:
            return None

        # Sort by filename (timestamp prefix at start ensures chronological order)
        candidates.sort()
        # If base hash only, pick latest; if full id, also last (only one usually)
        file_path = candidates[-1]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
            return CharacteristicsExtractionResult(**data)
        except Exception:
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

        # Log experiment start (include separation of hash/timestamp parts for clarity)
        log_file = self.results_saver.logs_dir / f"{experiment_id}_experiment.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Experiment {experiment_id} started at {datetime.now().isoformat()}\n")
            base_part = experiment_id.split('_')[0]
            f.write(f"Base config hash: {base_part}\n")
            f.write(f"Configuration: {json.dumps(asdict(config), indent=2)}\n")

        return experiment_id