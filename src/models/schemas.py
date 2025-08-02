from pydantic import BaseModel
from typing import Optional
from enum import Enum

class DetailLevel(str, Enum):
    """Enumeration for detail levels in descriptions."""
    NOT_FOUND = "Not Found"
    BASIC = "Basic"
    DETAILED = "Detailed"
    COMPREHENSIVE = "Comprehensive"

class DTCharacteristics(BaseModel):
    system_under_study: Optional[str]
    physical_acting_components: Optional[str]
    physical_sensing_components: Optional[str]
    physical_to_virtual_interaction: Optional[str]
    virtual_to_physical_interaction: Optional[str]
    dt_services: Optional[str]
    twinning_time_scale: Optional[str]
    multiplicities: Optional[str]
    life_cycle_stages: Optional[str]
    dt_models_and_data: Optional[str]
    tooling_and_enablers: Optional[str]
    dt_constellation: Optional[str]
    twinning_process_and_dt_evolution: Optional[str]
    fidelity_and_validity_considerations: Optional[str]
    dt_technical_connection: Optional[str]
    dt_hosting_deployment: Optional[str]
    insights_and_decision_making: Optional[str]
    horizontal_integration: Optional[str]
    data_ownership_and_privacy: Optional[str]
    standardization: Optional[str]
    security_and_safety_considerations: Optional[str]


# Group 1: Purpose
class Block1Characteristics(BaseModel):
    system_under_study: Optional[str]
    dt_services: Optional[str]
    tooling_and_enablers: Optional[str]

# Group 2: Orchestration
class Block2Characteristics(BaseModel):
    twinning_time_scale: Optional[str]
    multiplicities: Optional[str]
    dt_constellation: Optional[str]
    horizontal_integration: Optional[str]

# Group 3: Components
class Block3Characteristics(BaseModel):
    dt_models_and_data: Optional[str]
    physical_acting_components: Optional[str]
    physical_sensing_components: Optional[str]
    fidelity_and_validity_considerations: Optional[str]

# Group 4: Connectivity
class Block4Characteristics(BaseModel):
    physical_to_virtual_interaction: Optional[str]
    virtual_to_physical_interaction: Optional[str]
    dt_technical_connection: Optional[str]
    dt_hosting_deployment: Optional[str]

# Group 5: Lifecycle
class Block5Characteristics(BaseModel):
    life_cycle_stages: Optional[str]
    twinning_process_and_dt_evolution: Optional[str]
    insights_and_decision_making: Optional[str]
    standardization: Optional[str]

# Group 6: Governance
class Block6Characteristics(BaseModel):
    data_ownership_and_privacy: Optional[str]
    security_and_safety_considerations: Optional[str]
