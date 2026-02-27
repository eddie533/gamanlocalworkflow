#!/usr/bin/env python3
"""
City Extractor - Extract city name from a full address using GPT-5-nano.
"""

import json
from openai import OpenAI
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')


class CityExtractor:
    """Extract the city from a full address string using gpt-5-nano."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """You extract the city name from an address.

Given a full address, return ONLY the city/town name.

---
**Output requirements**
• Respond **only** with a valid JSON object.
• Do **not** wrap it in backticks or code fences.
• Use exactly this key:
  - "city" (the city/town name, or "" if not identifiable)"""

    def extract(self, address: str) -> str:
        """Return the city name from the given address string."""
        if not address or not str(address).strip():
            return ""

        user_prompt = f"Address: {str(address).strip()}"

        try:
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            response = self.client.responses.create(
                model="gpt-5-nano",
                input=combined_input,
            )

            result = json.loads(response.output_text)
            return result.get("city", "")

        except Exception as e:
            print(f"  error extracting city from '{address}': {e}")
            return ""
