from pathlib import Path
from typing import Dict, Any
from abc import ABC, abstractmethod

class IOMLWriter(ABC):
    """Interface for OML writing operations following ISP."""
    
    @abstractmethod
    def write_oml(self, oml_content: str, output_path: str) -> bool:
        """Write OML content to specified path."""
        pass

class OMLFileWriter(IOMLWriter):
    """Concrete implementation for writing OML to files following SRP."""
    
    def write_oml(self, oml_content: str, output_path: str = r"data\DTDF\src\oml\bentleyjoakes.github.io\LLM_described_DT\llm_dt.oml") -> bool:
        """
        Write OML content to the specified file path, wrapped in proper OML description structure.
        
        Args:
            oml_content: The OML content to write
            output_path: Path where the OML file should be written
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Wrap content in OML description structure
            wrapped_oml = self._wrap_in_oml_description(oml_content)
            
            # Write OML content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(wrapped_oml)
            
            print(f"✅ OML output written to: {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error writing OML file: {e}")
            return False
    
    def _wrap_in_oml_description(self, oml_content: str) -> str:
        """
        Wrap the OML content in the required description structure.
        """
        # Indent the content properly (4 spaces for each line)
        indented_content = '\n'.join(f"    {line}" if line.strip() else line 
                                    for line in oml_content.split('\n'))
        
        return f"""description <https://bentleyjoakes.github.io/LLM_described_DT/llm_dt#> as incubator {{

	uses <https://bentleyjoakes.github.io/DTDF/vocab/DTDFVocab#> as DTDFVocab
	
	uses <https://bentleyjoakes.github.io/DTDF/vocab/base#> as base
	extends <https://bentleyjoakes.github.io/DTDF/desc/baseDesc#> as baseDesc
	
{indented_content}

}}"""