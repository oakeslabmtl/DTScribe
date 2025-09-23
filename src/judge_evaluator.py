from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
import json
import re

class JudgeEvaluator:
    """LLM-as-a-Judge component to score extracted characteristics.
    Uses the same LLM instance as EnhancedRAGPipeline for consistency.
    """
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["characteristics", "retrieved_docs"],
            template=
            """
You are an expert evaluator in Digital Twin information extraction and ontology-based modeling.


Your task is to evaluate the quality of an extracted group of characteristics of a Digital Twin, based on the provided source text snippets that served as evidence for the extraction.


You must:
1. Read the extracted characteristic carefully.
2. Compare it to the evidence provided — only use the information contained in the source snippets. Do not rely on outside knowledge.
3. Score it on the 1–5 scale below.
4. Provide reasoning for your score, citing 1–3 short quotes from the sources that justify your decision. If no relevant evidence is present, explicitly say so.
5. Output in the required JSON format.


---


## Scoring Scale


1. Very poor
- "Not found" or states the characteristic is missing.
- Vague or generic information.
- Uses speculative language such as "likely", "possibly" instead of directly using the text.
- No relevance to the specific system in the sources.

2. Poor
- Very general definition.
- Not specific to the system mentioned in the source documents.
- Lacks technical content.

3. Fair
- Relevant to the specific system.
- Some correct context, but missing key technical details.

4. Good
- Correct and specific.
- Contains relevant technical details.
- Could still be expanded with more depth.

5. Very good
- Highly specific and detailed.
- Provides in-depth technical explanation of the characteristic's behavior within the system.
- Strongly grounded in the provided sources.


---


## Instructions to the Evaluator LLM
- Only use the provided source text snippets to determine accuracy.
- If the extraction includes unsupported claims or speculation, reduce the score accordingly.
- If no evidence supports the extraction or the extraction is "Not Found", assign a 1.
- Always explain your reasoning before giving the score.


---


### Output Format (JSON)


Return an array where each element corresponds to one characteristic evaluated.


Each element must have:
- "characteristic": the name of the characteristic evaluated
- "reasoning": your explanation for the score, citing short quotes from the sources
- "score": integer from 1 to 5


Example:
[
    {{
        "characteristic": "System under study",
        "reasoning": "Mentions the incubator and its role; specific, but lacks behavioral detail. Supported by quotes: 'The incubator maintains optimal fermentation temperature'.",
        "score": 4
    }},
    {{
        "characteristic": "Physical acting components",
        "reasoning": "Lists fan and heater, describes control methods; detailed and supported by text.",
        "score": 5
    }}
]
---


Extracted Characteristic:
{characteristics}


Source Evidence:
{retrieved_docs}
---
"""
        )

    def _clean_response(self, text: str) -> str:
        # remove <think>...</think> blocka
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # remove ```json ... ``` or ```
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        return cleaned

    
    def _coerce_to_array(self, text: str) -> str:
        txt = text.strip()
        if txt.startswith("{") and txt.endswith("}"):
            return f"[{txt}]"
        
        start = txt.find("[")
        end = txt.rfind("]")
        if start != -1 and end != -1:
            return txt[start:end+1]
        return txt
    

    def evaluate(self, extracted: Dict[str, Any], docs: List[Any]) -> List[Dict[str, Any]]:
    
        docs_content = "\n\n".join([
            f"# Document {i+1}:\n{getattr(doc, 'page_content', str(doc))}" 
            for i, doc in enumerate(docs)
        ])
        
        characteristics_text = json.dumps(extracted, indent=2)

        # print(f"JudgeEvaluator.evaluate(): Characteristics to evaluate: {characteristics_text}")

        prompt = self.prompt.format(
            characteristics=characteristics_text,
            retrieved_docs=docs_content
        )

        # print(f"JudgeEvaluator.evaluate(): Generated prompt for LLM:\n{prompt}")
        
        response = self.llm.invoke(prompt)
        # print(f"JudgeEvaluator.evaluate(): LLM raw response: {response}")

        raw_text = getattr(response, "content", str(response))

        # print(f"JudgeEvaluator.evaluate(): LLM output text: {raw_text}")

        cleaned = self._clean_response(raw_text)
        cleaned = self._coerce_to_array(cleaned)

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                parsed = [parsed]

        except Exception:
            parsed = [{
                "characteristic": "ALL_BLOCK",
                "reasoning": "Could not parse judge output; treating as low confidence",
                "score": 1
            }]
        return parsed

