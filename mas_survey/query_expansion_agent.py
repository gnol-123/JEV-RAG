from dataclasses import dataclass
from typing import List, Optional
import re
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

from utils.llm_client import LLMClient


@dataclass
class QueryExpansionAgent:

    client: Optional[LLMClient] = None

    def __init__(self):

        # ---------------- Question words keep if at start ---------------------- #
        self.question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which'}

        # ---------------- Initialize LLM client ---------------------- #
        try:
            self.client = LLMClient()
        except:
            self.client = None


    # LLM QUERY EXPANSION -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def llm_expand_query(self, query: str, distribution: dict = None, max_variations: int = 3) -> List[str]:

        sys_prompt = """
        You are a query optimization expert for document retrieval systems.

        Your task is to analyze a survey question and extract the most important keywords and concepts
        that would help find relevant social media posts and documents.

        INSTRUCTIONS:
        1. Identify the CORE TOPIC of the query (3-6 key terms)
        2. Extract any SPECIFIC ENTITIES or CONCEPTS mentioned
        3. Remove filler words, question words, and unnecessary phrases
        4. Generate 2-3 focused search variations that capture different aspects
        5. If any word comes before \\ example: Middle east. Try to include atleast two versions of the term

        GUIDELINES:
        - Focus on nouns, verbs, and specific concepts
        - Keep queries SHORT (3-6 words each)
        - Each variation should emphasize different aspects of the topic
        - Avoid generic words like "people", "think", "feel", "opinion"
        - ***STRICTLY AVOID RETURNING UNIFORM DISTRIBUTIONS like 0.3333333, 0.333333, 0.333334***

        Return a JSON object with:
        {
            "core_terms": ["term1", "term2", "term3"],
            "variations": ["query1", "query2", "query3", "query4"...]
        }
        """

        usr_prompt = f"""
        Survey Question: {query}
        Survey Answers: {distribution}

        Extract the core terms and generate focused search query variations.
        """

        try:
            result = self.client.chat_json(
                system=sys_prompt,
                user=usr_prompt,
                max_tokens=300,
                temperature=0.3
            )

            if "_error" not in result:
                variations = []

                # ---------------- Add original query first ---------------------- #
                variations.append(query)

                # ---------------- Add LLM variations ---------------------- #
                llm_variations = result.get("variations", [])
                for var in llm_variations[:max_variations]:
                    if var and var not in variations:
                        variations.append(var)

                # ---------------- Add core terms if no variations ---------------------- #
                core_terms = result.get("core_terms", [])
                if core_terms:
                    core_query = " ".join(core_terms[:4])
                    if core_query not in variations:
                        variations.append(core_query)

                return variations[:max_variations + 1]

        except Exception as e:
            print(f"Warning: LLM expansion failed: {e}")

        # ---------------- Fallback to original query ---------------------- #
        return [query]


    # BASIC QUERY EXPANSION -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def basic_expand_query(self, query: str, max_variations: int = 3) -> List[str]:

        variations = []

        query = query.strip()
        if not query:
            return [query]

        variations.append(query)

        # ---------------- Remove stop words ---------------------- #
        stop_words = set(stopwords.words('english'))
        words = query.split()
        filtered = []

        for i, word in enumerate(words):
            word_lower = word.lower()

            if i == 0 and word_lower in self.question_words:
                continue

            if word_lower not in stop_words:
                filtered.append(word)

        filtered_query = " ".join(filtered)
        if filtered_query and filtered_query != query:
            variations.append(filtered_query)

        return variations[:max_variations]


    # MAIN EXPAND QUERY -------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def expand_query(self, query: str, distribution: dict = None, max_variations: int = 5, use_llm: bool = True) -> List[str]:

        # ---------------- Do LLM expansion if client ---------------------- #
        if use_llm and self.client is not None:
            return self.llm_expand_query(query, distribution, max_variations=max_variations)

        # ---------------- Fall back to basic expansion ---------------------- #
        return self.basic_expand_query(query, max_variations=max_variations)
