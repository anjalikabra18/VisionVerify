import re
from typing import List, Dict

def extract_claims(transcript: str) -> List[str]:
    """Extract potential claims from video transcript."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', transcript)
    return [sentence.strip() for sentence in sentences if sentence]

def fact_check_claims(claims: List[str], agent) -> Dict[str, str]:
    """Perform fact-checking for a list of claims using web search."""
    results = {}
    for claim in claims:
        query = f"Verify the following claim: {claim}"
        try:
            response = agent.run(query)
            results[claim] = response.content
        except Exception as e:
            results[claim] = f"Error verifying claim: {str(e)}"
    return results
