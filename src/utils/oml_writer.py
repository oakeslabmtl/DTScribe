from pathlib import Path
from typing import Dict, Any
import re
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
        print("💾 Writing OML to file...")
        try:
            # Ensure directory exists
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Wrap content in OML description structure
            wrapped_oml = self._wrap_in_oml_description(oml_content)
            
            # Write OML content
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(wrapped_oml)
            print(f"📝 OML output written to: {output_path}")
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
        
        return f"""description <https://bentleyjoakes.github.io/LLM_described_DT/llm_dt#> as llm_dt {{

	uses <https://bentleyjoakes.github.io/DTDF/vocab/DTDFVocab#> as DTDFVocab
	
	uses <https://bentleyjoakes.github.io/DTDF/vocab/base#> as base
	extends <https://bentleyjoakes.github.io/DTDF/desc/baseDesc#> as baseDesc
	
{indented_content}

}}"""
    
    def _combine_oml_with_validation_errors(self, oml_content: str, validation_errors: str) -> str:
        """
        Combine OML content with validation errors.
        """
        # Parse validation errors of the form: "[10, 9]: Couldn't resolve ..."
        # Group messages by line number (1-based)
        line_errors: Dict[int, list[str]] = {}
        if validation_errors:
            for raw in validation_errors.splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                m = re.match(r"^\[(\d+),\s*(\d+)\]:\s*(.+)$", raw)
                if not m:
                    # If the line doesn't match expected pattern, attach it as a general note to line 1
                    line_errors.setdefault(1, []).append(raw)
                    continue
                line_num = int(m.group(1)) - 6  # Adjust index for header lines
                col_num = int(m.group(2))
                message = m.group(3).strip()
                # Compose concise inline comment keeping column info
                comment = f"TODO: Fix the error on this line helped with this error message: '{message}'"
                line_errors.setdefault(line_num, []).append(comment)

        # Append errors as inline comments at the end of their respective lines
        if not line_errors:
            return oml_content

        lines = oml_content.splitlines(keepends=True)
        annotated_lines: list[str] = []
        for idx, line in enumerate(lines, start=1):
            # Preserve original line ending
            newline = ""
            core = line
            if line.endswith("\r\n"):
                newline = "\r\n"
                core = line[:-2]
            elif line.endswith("\n"):
                newline = "\n"
                core = line[:-1]
            elif line.endswith("\r"):
                newline = "\r"
                core = line[:-1]

            if idx in line_errors:
                joined = " | ".join(line_errors[idx])
                core = f"{core} // {joined}"

            annotated_lines.append(core + newline)

        return "".join(annotated_lines)