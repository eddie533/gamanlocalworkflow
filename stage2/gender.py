#!/usr/bin/env python3
"""
Gender Guesser - Guess contact gender from first name using GPT-5-nano.
"""

import json
from openai import OpenAI
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')


class GenderGuesser:
    """Guess gender from a first name using gpt-5-nano (cheapest model, no web search)."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """You are a name-to-gender classifier.
Given a first name, guess the most likely gender.

---
**Output requirements**
• Respond **only** with a valid JSON object.
• Do **not** wrap it in backticks or code fences.
• Use exactly this key:
  - "gender" ("Male", "Female", or "Unknown")"""

    def guess(self, first_name: str) -> str:
        """Return 'Male', 'Female', or 'Unknown' for the given first name."""
        if not first_name or not first_name.strip():
            return "Unknown"

        user_prompt = f"First name: {first_name.strip()}"

        try:
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            response = self.client.responses.create(
                model="gpt-5-nano",
                input=combined_input,
            )

            result = json.loads(response.output_text)
            gender = result.get("gender", "Unknown")

            if gender not in ("Male", "Female", "Unknown"):
                return "Unknown"

            return gender

        except Exception as e:
            print(f"  error guessing gender for '{first_name}': {e}")
            return "Unknown"
