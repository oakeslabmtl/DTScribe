"""
Enhanced Pydantic schemas with better validation and field descriptions for Digital Twin characteristics.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum
import re


class DetailLevel(str, Enum):
    """Enumeration for detail levels in descriptions."""
    NOT_FOUND = "Not Found"
    BASIC = "Basic"
    DETAILED = "Detailed"
    COMPREHENSIVE = "Comprehensive"


class DTCharacteristics(BaseModel):
    """Complete Digital Twin characteristics with enhanced validation."""
    
    system_under_study: Optional[str] = Field(
        None,
        description="Comprehensive description of the Physical Twin system including type, components, operational domain, and key characteristics"
    )
    physical_acting_components: Optional[str] = Field(
        None,
        description="Detailed description of actuators and control mechanisms including types, capabilities, control interfaces, and safety constraints"
    )
    physical_sensing_components: Optional[str] = Field(
        None,
        description="Comprehensive description of sensors and measurement systems including types, specifications, spatial distribution, and data collection capabilities"
    )
    physical_to_virtual_interaction: Optional[str] = Field(
        None,
        description="Detailed description of data flows from physical to digital including protocols, data types, transmission frequencies, and quality assurance"
    )
    virtual_to_physical_interaction: Optional[str] = Field(
        None,
        description="Comprehensive description of control flows from digital to physical including command types, validation mechanisms, and feedback systems"
    )
    dt_services: Optional[str] = Field(
        None,
        description="Detailed description of Digital Twin services including types, target users, capabilities, and service interactions"
    )
    twinning_time_scale: Optional[str] = Field(
        None,
        description="Comprehensive description of temporal aspects including synchronization frequencies, latency requirements, and time scales"
    )
    multiplicities: Optional[str] = Field(
        None,
        description="Detailed description of system multiplicities including multiple twin instances, hierarchical structures, and coordination mechanisms"
    )
    life_cycle_stages: Optional[str] = Field(
        None,
        description="Comprehensive description of lifecycle phases including development stages, representation types, and transition management"
    )
    dt_models_and_data: Optional[str] = Field(
        None,
        description="Detailed description of models and data including types, sources, relationships, validation approaches, and management strategies"
    )
    tooling_and_enablers: Optional[str] = Field(
        None,
        description="Comprehensive description of tools and technologies including platforms, frameworks, development tools, and their specific functionalities"
    )
    dt_constellation: Optional[str] = Field(
        None,
        description="Detailed description of system orchestration including architecture patterns, component coordination, and resource management"
    )
    twinning_process_and_dt_evolution: Optional[str] = Field(
        None,
        description="Comprehensive description of engineering processes including development methodologies, quality assurance, and evolution management"
    )
    fidelity_and_validity_considerations: Optional[str] = Field(
        None,
        description="Detailed description of fidelity requirements, validation methods, uncertainty quantification, and quality assurance processes"
    )
    dt_technical_connection: Optional[str] = Field(
        None,
        description="Comprehensive description of network infrastructure including protocols, topology, requirements, and reliability mechanisms"
    )
    dt_hosting_deployment: Optional[str] = Field(
        None,
        description="Detailed description of deployment infrastructure including hosting platforms, resource requirements, and scalability considerations"
    )
    insights_and_decision_making: Optional[str] = Field(
        None,
        description="Comprehensive description of decision support capabilities including insight types, communication methods, and confidence measures"
    )
    horizontal_integration: Optional[str] = Field(
        None,
        description="Detailed description of external system integration including protocols, data exchange, and interoperability standards"
    )
    data_ownership_and_privacy: Optional[str] = Field(
        None,
        description="Comprehensive description of data governance including ownership policies, privacy compliance, and consent management"
    )
    standardization: Optional[str] = Field(
        None,
        description="Detailed description of standards compliance including industry standards, specifications, and certification requirements"
    )
    security_and_safety_considerations: Optional[str] = Field(
        None,
        description="Comprehensive description of security and safety measures including threat mitigation, access control, and fail-safe mechanisms"
    )
    
    @field_validator('*', mode='before')
    @classmethod
    def validate_not_empty(cls, v):
        """Ensure fields are not empty strings."""
        if isinstance(v, str) and v.strip() == "":
            return "Not Found"
        return v

# Validation utilities
class ExtractionQualityValidator:
    """Utility class for validating extraction quality."""
    
    @staticmethod
    def validate_technical_content(text: str) -> Dict[str, Any]:
        """Validate if text contains sufficient technical content."""
        if not text or text == "Not Found":
            return {"is_valid": False, "reason": "No content found"}
        
        # Check for technical indicators
        technical_indicators = [
            r'\b\d+(\.\d+)?\s*(Hz|kHz|MHz|GHz)\b',  # Frequencies
            r'\b\d+(\.\d+)?\s*(ms|μs|ns|s)\b',      # Time units
            r'\b\d+(\.\d+)?\s*(°C|K|F)\b',          # Temperature
            r'\b\d+(\.\d+)?\s*(Mbps|Gbps|MB/s)\b', # Data rates
            r'\b(API|REST|MQTT|OPC UA|HTTP|TCP|UDP)\b',  # Protocols
            r'\b(cloud|edge|on-premise|hybrid)\b',  # Deployment types
        ]
        
        technical_score = sum(1 for pattern in technical_indicators 
                            if re.search(pattern, text, re.IGNORECASE))
        
        return {
            "is_valid": len(text) >= 50 and technical_score > 0,
            "technical_score": technical_score,
            "length": len(text),
            "reason": f"Technical score: {technical_score}, Length: {len(text)}"
        }
    
    @staticmethod
    def assess_extraction_completeness(characteristics: BaseModel) -> Dict[str, Any]:
        """Assess the completeness and quality of extracted characteristics."""
        data = characteristics.model_dump(exclude_none=True)
        
        total_fields = len(characteristics.model_fields)
        extracted_fields = len([v for v in data.values() if v != "Not Found"])
        
        quality_scores = []
        for field, value in data.items():
            if value != "Not Found":
                validation = ExtractionQualityValidator.validate_technical_content(value)
                quality_scores.append(validation["technical_score"])
        
        return {
            "completeness_rate": (extracted_fields / total_fields) * 100,
            "average_technical_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "total_fields": total_fields,
            "extracted_fields": extracted_fields,
            "high_quality_fields": len([s for s in quality_scores if s >= 2])
        }
