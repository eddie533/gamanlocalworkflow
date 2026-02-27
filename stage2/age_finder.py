#!/usr/bin/env python3
"""
Age Finder - Estimate a contact's age range using GPT-5 with web search.
"""

import json
from openai import OpenAI
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

ALLOWED_AGE_RANGES = ["<40", "40-50", "50-60", "60+"]


class AgeFinder:
    """Estimate age range from a person's name, company, and role using GPT-5."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = f"""You estimate a person's age range based on their name, company, and role.

Use web search to find information about the person, then estimate their age range.

You MUST select EXACTLY ONE age range from this list:
{', '.join(ALLOWED_AGE_RANGES)}

---
**Output requirements**
• Respond **only** with a valid JSON object.
• Do **not** wrap it in backticks or code fences.
• Use exactly this key:
  - "age" (one of: "<40", "40-50", "50-60", "60+")"""

    def find(self, first_name: str, last_name: str, company: str, role: str) -> str:
        """Return an age range string."""
        if not first_name or not first_name.strip():
            return ""

        parts = []
        if first_name:
            parts.append(f"First name: {first_name.strip()}")
        if last_name:
            parts.append(f"Last name: {last_name.strip()}")
        if company:
            parts.append(f"Company: {str(company).strip()}")
        if role:
            parts.append(f"Role: {str(role).strip()}")

        user_prompt = "\n".join(parts)

        try:
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            response = self.client.responses.create(
                model="gpt-5",
                input=combined_input,
                tools=[{"type": "web_search"}],
            )

            result = json.loads(response.output_text)
            age = result.get("age", "")

            if age not in ALLOWED_AGE_RANGES:
                return ""

            return age

        except Exception as e:
            print(f"  error finding age for '{first_name} {last_name}': {e}")
            return ""
