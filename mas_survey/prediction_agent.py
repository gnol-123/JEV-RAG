from dataclasses import dataclass
from typing import Dict, List, Optional

from utils.llm_client import LLMClient
from mas_survey.retreival_agent import RetrievedDoc


@dataclass
class PredictionAgent:

    client: Optional[LLMClient] = None

    def predict_distribution(self, query: str, answer_options: List[str],
                             retrieved_docs: List[RetrievedDoc]) -> Dict[str, float]:

        # ---------------- Initialize client ---------------------- #
        if self.client is None:
            self.client = LLMClient()

        # ---------------- No docs fallback ---------------------- #
        if not retrieved_docs or not answer_options:
            uniform = 1.0 / len(answer_options) if answer_options else 0.0
            print("Warning: No docs or no answer options. Defaulting to uniform dist")
            return {opt: uniform for opt in answer_options}

        # ---------------- Build context from top 15 docs ---------------------- #
        ctx_parts = []

        for i, doc in enumerate(retrieved_docs[:15]):
            txt = " ".join([
                doc.title,
                doc.description,
                doc.post_content,
                doc.content if doc.content else ""
            ]).strip()

            if txt:
                ctx_parts.append(f"[Doc {i}] {txt[:250]}...")

        ctx = "\n".join(ctx_parts) if ctx_parts else "Limited context available."

        # ---------------- Build prompts ---------------------- #
        option_list = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(answer_options)])

        sys_prompt = """
                    You are analyzing social media discussions to predict survey responses.

                    CRITICAL INSTRUCTIONS:
                    1. COUNT how many documents support each answer option
                    2. WEIGHT by sentiment strength (strong opinions count more)
                    3. TRANSLATE document patterns into population probabilities
                    4. DO NOT default to uniform distributions - make real predictions!
                    5. AVOID GIVING EQUAL UNIFORM DISTRIBUTION UNLESS THE SENTIMENT IS COMPLETELY EQUAL
                    6. PROVIDE PROBABILITIES WITH 6^-10 DECIMAL PLACES OF PRECISION

                    Analysis Process:
                    - If 8/10 documents support Option A then Option A should have 0.60-0.80 probability
                    - If documents are split 5/3/2, reflect that in probabilities (e.g. 0.50/0.30/0.20)
                    - Strong negative sentiment toward an option, lower probability (0.05-0.15)
                    - No mentions of an option, very low probability (0.02-0.05)

                    Return ONLY a valid JSON object with probabilities for each option. Probabilities must:
                    - Be between 0 and 1
                    - Have 10^-6 decimal places (e.g., 0.3571428571, not 0.35)
                    - Sum to exactly 1.0
                    - Reflect the sentiment and opinions in the provided documents

                    Format: {"Option 1": 0.2500000000, "Option 2": 0.5000000000, "Option 3": 0.2500000000}
                    """

        usr_prompt = f"""
        Query: {query}

        Answer Options: {option_list}

        Social Media Context:{ctx}

        Based on the sentiment and opinions in these posts, predict the probability distribution over the answer options.
        """

        try:
            # ---------------- Query LLM ---------------------- #
            result = self.client.chat_json(
                system=sys_prompt,
                user=usr_prompt,
                schema_hint='{\n' + ',\n'.join([f'  "{opt}": 0.0' for opt in answer_options]) + '\n}',
                max_tokens=300,
                temperature=0.3
            )

            # --------------------- Parse distribution --------------- #
            if "_error" not in result:
                dist = {}
                total = 0

                for opt in answer_options:
                    prob = result.get(opt, 0.0)
                    if isinstance(prob, (int, float)) and prob >= 0:
                        dist[opt] = float(prob)
                        total += dist[opt]
                    else:
                        dist[opt] = 0.0

                # ---------------- Normalize ---------------------- #
                if total > 0:
                    dist = {k: v / total for k, v in dist.items()}
                    return dist

        except:
            print("Failed to get prediction from LLM. Defaulting to uniform distribution.")

        # ---------------- Fallback ---------------------- #
        uniform = 1.0 / len(answer_options)
        return {opt: uniform for opt in answer_options}
