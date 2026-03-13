import csv
import json
import time
from openai import OpenAI
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file (check parent directories)
load_dotenv(Path(__file__).parent.parent / '.env')

class CompanyEnricher:
    """Enrich companies with vertical classification and sub-vertical descriptions"""

    ALLOWED_VERTICALS = [
        "Agriculture", "Compliance", "Construction", "CyberSecurity", "Defence",
        "Digital Infrastructure", "Education", "Energy", "Engineering",
        "Financial Services", "Food & Beverages", "Gaming", "Geomatics",
        "Healthcare", "Cross-industry", "Hospitality", "Information Services",
        "Infrastructure", "IT Services", "Legal", "Lockers", "Logistics",
        "Manufacturing", "Maritime", "Media", "Media Marketing",
        "Public Sector", "Retail", "Scientific", "Sustainability",
        "Testing & Inspection", "Transport vehicles", "Transportation",
        "Utilities", "Specialised Industry", "Financial Operations Services",
        "Outsourced Operations"
    ]

    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 4500):
        """Initialize the enricher with OpenAI client

        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
            requests_per_minute: Max requests per minute (default 4500 to stay under 5000 RPM limit)
        """
        # Increase timeout for flex processing (default is 10 min, set to 15 min)
        self.client = OpenAI(
            api_key=api_key,
            timeout=900.0  # 15 minutes for flex processing
        )
        self.system_prompt = f"""You are a B2B software market analyst specializing in vertical market classification and company positioning.

Your task is to analyze companies and:
1. Assign the ONE best-fit vertical based on their primary customer end-market
2. Create a professional sub-vertical description for outreach
3. Identify how the company refers to themselves (informal name)

VERTICAL CLASSIFICATION RULES:
- Choose from this EXACT list (spelling matters): {', '.join(self.ALLOWED_VERTICALS)}
- ⚠️ Note that there is not a "Real Estate" vertical on the list — classify Real Estate as "Construction"
- ⚠️ Note that "Horizontal" and "Vertical SW" are NOT valid verticals. Use "Cross-industry" and "Specialised Industry" instead.

**Classification Hierarchy (apply in this order — order matters!):**

1. **Industry Vertical**
   - First check if the company primarily serves a specific industry
   - If most customers belong to one sector, classify under that vertical
   - Consider who BUYS the software, not what industry the company is in
   - Examples:
     • Healthcare software → "Healthcare"
     • Financial services platforms → "Financial Services"
     • Manufacturing solutions → "Manufacturing"
     • Logistics platforms → "Logistics"
     • Construction/Real Estate → "Construction"

2. **Outsourced Operations**
   - If the company primarily executes operational processes for clients on an ongoing basis (i.e. outsourced workflows or managed operations), classify as "Outsourced Operations"
   - ⚠️ This must be checked BEFORE Cross-industry, otherwise many operational services will be misclassified
   - Typical examples:
     • Outsourced financial operations
     • Compliance monitoring services
     • Operational back-office services
     • Recurring operational workflow execution for clients

3. **Cross-industry**
   - If the company provides technology or services used across many industries and is NOT primarily an outsourced operational execution provider, classify as "Cross-industry"
   - Examples:
     • Cybersecurity platforms (if truly cross-industry, otherwise use "CyberSecurity")
     • Data analytics platforms
     • Digital infrastructure monitoring tools
     • Information services providers
     • General business software (CRM, HR, productivity, collaboration)

4. **Specialised Industry**
   - If the company operates in a narrow operational niche that does not map cleanly to an industry vertical, classify as "Specialised Industry"
   - Examples:
     • Smart locker platforms
     • Laboratory workflow software
     • Specialised inspection technologies

**Simple decision rule:**
- Serves a specific industry → assign that industry vertical
- Executes outsourced operational workflows → "Outsourced Operations"
- Used across many industries → "Cross-industry"
- Niche operational domain → "Specialised Industry"

SUB-VERTICAL DESCRIPTION:
- Should complete: "I noticed [Company Name] as a leader in <subvertical>."
- Keep it professional and outreach-ready
- Use "and" instead of "&" symbols
- Natural, concise phrasing (3-8 words typically)

LANGUAGE-SPECIFIC RULES:
- **French companies (Country: France/FR):**
  • Write subvertical in French
  • Include proper French article: "le", "la", "les", or "l'" before the noun
  • Example: "la gestion de la chaîne logistique" or "les solutions de cybersécurité"

- **Spanish companies (Country: Spain/ES):**
  • Write subvertical in Spanish
  • Include proper Spanish article: "el", "la", "los", or "las" before the noun
  • Example: "la gestión de la cadena de suministro" or "las soluciones de ciberseguridad"

- **All other countries:** Write subvertical in British English

INFORMAL NAME:
- How the company refers to itself on their website/marketing
- Often the short form without legal suffixes (SAS, LLC, etc.)
- Check their website branding"""

        # Rate limiting - DISABLED: OpenAI API has built-in rate limiting
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_delay:
            sleep_time = self.min_delay - time_since_last_request
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def enrich_company(self, company_data: Dict) -> Dict:
        """Enrich a single company with vertical classification"""
        # Rate limiting - DISABLED: OpenAI API has built-in rate limiting
        # self._rate_limit()

        # Prepare the enrichment prompt
        country = company_data.get('Country', 'Unknown')

        user_prompt = f"""Analyze the following company:

Company Name: {company_data.get('Company Name', 'Unknown')}
Country: {country}
Informal Name: {company_data.get('Informal Name', 'Unknown')}
Website: {company_data.get('Website', 'Not provided')}
Description: {company_data.get('Description', 'No description')}
Specialties: {company_data.get('Specialties', 'None listed')}
Products/Services: {company_data.get('Products and Services', 'None listed')}
End Markets: {company_data.get('End Markets', 'None listed')}
Industries: {company_data.get('Industries', 'Unknown')}

Classify this company and provide enrichment data.

⚠️ IMPORTANT: Check the Country field to determine the language for the subvertical:
- If Country is "France" or "FR" → subvertical in French with article (le/la/les/l')
- If Country is "Spain" or "ES" → subvertical in Spanish with article (el/la/los/las)
- Otherwise → subvertical in British English

---
**Output requirements**
• Respond **only** with a single, valid JSON object.
• Do **not** wrap it in backticks, code fences, or add any extra text.
• Use exactly these keys:
  - "vertical" (ONE value from the allowed list - exact spelling)
  - "subvertical" (professional phrase in appropriate language based on country - see language rules above)
  - "informal_name" (how they refer to themselves on website/branding)
• Any deviation will break downstream parsing."""

        try:
            # Combine system and user prompts for responses API
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            # Use Responses API with flex processing for cost savings
            # Retry logic for 429 Resource Unavailable errors
            max_retries = 3
            retry_delay = 5  # seconds

            used_flex = False
            for attempt in range(max_retries):
                try:
                    response = self.client.responses.create(
                        model="gpt-5",  # Using GPT-5 with 1M TPM and 5K RPM limits
                        input=combined_input,
                        tools=[{"type": "web_search"}],  # Enable web search for accurate classification
                        service_tier="flex"  # Use flex processing for lower costs
                    )
                    used_flex = True
                    break

                except Exception as e:
                    error_msg = str(e)

                    # Check if it's a 429 Resource Unavailable error
                    if "429" in error_msg and "Resource Unavailable" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            time.sleep(wait_time)
                        else:
                            response = self.client.responses.create(
                                model="gpt-5",
                                input=combined_input,
                                tools=[{"type": "web_search"}]
                                # No service_tier = standard processing
                            )
                            used_flex = False
                            break
                    else:
                        # Different error, re-raise
                        raise

            result = json.loads(response.output_text)

            # Validate vertical is in allowed list
            vertical = result.get('vertical', 'Cross-industry')
            if vertical not in self.ALLOWED_VERTICALS:
                vertical = 'Cross-industry'

            return {
                "vertical": vertical,
                "subvertical": result.get('subvertical', ''),
                "informal_name": result.get('informal_name', company_data.get('Informal Name', '')),
                "company_name": company_data.get('Company Name', 'Unknown'),
                "company_data": company_data
            }

        except Exception as e:
            return {
                "vertical": "Cross-industry",
                "subvertical": "business solutions",
                "informal_name": company_data.get('Informal Name', company_data.get('Company Name', 'Unknown')),
                "error": str(e),
                "company_name": company_data.get('Company Name', 'Unknown'),
                "company_data": company_data
            }

    def process_csv(self, input_path: str, output_path: str = None):
        """Process all companies from CSV and enrich them"""
        input_file = Path(input_path)
        if output_path is None:
            output_path = input_file.parent.parent / "2enrich" / "enriched_companies.csv"

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        enriched_companies = []
        enrichment_results = []

        # Read CSV and process each company
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('Company Name'):
                    result = self.enrich_company(row)
                    enrichment_results.append(result)

                    enriched_row = row.copy()
                    enriched_row['Vertical'] = result.get('vertical', 'Cross-industry')
                    enriched_row['SubVertical'] = result.get('subvertical', '')
                    enriched_row['Informal Name (Enriched)'] = result.get('informal_name', '')
                    enriched_companies.append(enriched_row)

        # Save enriched companies to CSV
        if enriched_companies:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=enriched_companies[0].keys())
                writer.writeheader()
                writer.writerows(enriched_companies)

        results_path = output_file.parent / "enrichment_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(enrichment_results, f, indent=2)

        self.generate_summary(enrichment_results, output_file.parent / "enrichment_summary.txt")

        return enriched_companies, enrichment_results

    def generate_summary(self, results, output_path):
        """Generate a summary of vertical distribution"""
        vertical_counts = {}
        for result in results:
            if 'error' not in result:
                vertical = result.get('vertical', 'Cross-industry')
                vertical_counts[vertical] = vertical_counts.get(vertical, 0) + 1

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=== ENRICHMENT SUMMARY ===\n\n")
            f.write(f"Total companies enriched: {len(results)}\n\n")
            f.write("Vertical Distribution:\n")
            for vertical, count in sorted(vertical_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {vertical}: {count} companies\n")



def main():
    enricher = CompanyEnricher()
    input_csv = "../1validate/founder_owned_companies.csv"
    output_csv = "enriched_companies.csv"
    _, results = enricher.process_csv(input_csv, output_csv)
    print(f"Done — enriched {len(results)} companies → {output_csv}")


if __name__ == "__main__":
    main()
