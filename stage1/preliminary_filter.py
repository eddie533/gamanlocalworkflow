#!/usr/bin/env python3
"""
Preliminary Company Filter - Fast elimination of obvious non-software companies

Uses cheapest AI model (gpt-4o-mini) WITHOUT web search for quick filtering.
Eliminates consulting companies, agencies, and obvious non-software businesses
BEFORE the more expensive detailed business model validation.

This saves costs by filtering out ~50-70% of invalid companies early.
"""

import csv
import json
import time
from openai import OpenAI
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file (check parent directories)
load_dotenv(Path(__file__).parent.parent / '.env')

class PreliminaryFilter:
    def __init__(self, api_key: Optional[str] = None, requests_per_minute: int = 9000):
        """Initialize the filter with OpenAI client using cheapest model

        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
            requests_per_minute: Max requests per minute (default 9000 - gpt-4o-mini is very fast)
        """
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """You are a preliminary filter for B2B software companies. Your job is to eliminate companies that are CLEARLY not software companies.

⚠️ CRITICAL: Only eliminate companies that CLEARLY fit these categories. When in doubt, ALWAYS keep the company.

⚠️ IMPORTANT: If a company has NO description provided, it will be automatically kept (this function won't even be called). You will ONLY see companies WITH descriptions.

ELIMINATE (return "No") ONLY if the company CLEARLY fits one of these categories:

**1. Marketing/Advertising Agencies:**
- Traditional marketing/advertising/PR agencies (WITHOUT software products)
- Digital marketing agencies (WITHOUT software products)
- Creative/brand agencies
- Social media marketing agencies

⚠️ KEEP if they have software products (martech platforms, etc.)

**2. Project-Based Consulting/IT Services:**
- IT consulting firms (project-based)
- Management consulting firms
- Custom development shops (building software for clients)
- Business consulting firms

⚠️ KEEP if they have proprietary software products

**3. System Integrators/Implementation Partners:**
- SAP/Oracle/Salesforce/Microsoft implementation partners
- System integrators (implementing third-party software)
- Software resellers WITHOUT proprietary products

Examples: "SAP partner", "Oracle implementation", "Microsoft partner"

**4. Engineering/Technical Consulting:**
- Engineering consultancies (MEP, maritime, energy, defense)
- Technical design/architecture firms
- Project engineering firms

Examples: "engineering consultancy", "MEP design", "maritime engineering"

⚠️ KEEP if they have proprietary engineering software products

**5. Manufacturing/Hardware Companies:**
- Industrial equipment manufacturers
- Electronic device manufacturers
- Hardware product companies (WITHOUT significant SaaS)

Examples: "manufacturing equipment", "industrial machinery", "hardware products"

⚠️ KEEP if they have significant SaaS/software components

**6. Testing/QA Services Firms:**
- Software testing companies (project-based)
- Quality assurance services (project-based)
- Test engineering services

Examples: "testing services", "QA services", "quality engineering services"

⚠️ KEEP if they have proprietary testing platforms/SaaS

**7. Distributors/Resellers/Retailers:**
- Hardware/equipment distributors
- IT equipment resellers
- Retail/wholesale companies

Examples: "distributor of", "reseller", "wholesale", "retail"

**8. BPO/Outsourcing/Contact Centers:**
- Business process outsourcing
- Contact center services
- Customer service outsourcing

Examples: "BPO", "contact center", "call center", "outsourcing services"

⚠️ KEEP if they have proprietary CX/contact center software

**9. Logistics/Transportation:**
- Logistics companies
- Transportation services
- Freight/shipping companies

Examples: "logistics services", "transportation", "freight forwarding"

⚠️ KEEP if they have proprietary logistics software platforms

**10. Training/Coaching/HR Services:**
- Training/coaching firms (project-based)
- Leadership development
- HR consulting services

Examples: "training services", "leadership development", "coaching"

⚠️ KEEP if they have learning management software/platforms

**11. Building/Facilities/Infrastructure Services:**
- Building installation companies
- Facilities management services
- Construction/infrastructure firms

Examples: "building installations", "facilities management", "construction"

**12. Traditional Services (No Tech):**
- Insurance companies
- Financial services (non-tech)
- Publications/media (WITHOUT software products)

KEEP (return "Yes") for EVERYTHING ELSE:
- Any company with proprietary software products
- SaaS companies
- Platform businesses
- Tech-enabled services with software
- Data/analytics platforms
- Any company that might have software products
- ANY company that doesn't CLEARLY fit the categories above

⚠️ DEFAULT: When in doubt, KEEP the company. It's better to keep 100 uncertain companies than eliminate 1 valid software company."""

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

    def filter_company(self, company_data: Dict) -> Dict:
        """Quick filter to eliminate obvious non-software companies"""
        company_name = company_data.get('Company Name', 'Unknown')

        # Rate limiting - DISABLED: OpenAI API has built-in rate limiting
        # self._rate_limit()

        # Build context from available data - emphasize description
        context_parts = [f"Company: {company_name}"]

        if company_data.get('Website'):
            context_parts.append(f"Website: {company_data.get('Website')}")

        # Description is critical for preliminary filtering
        description = company_data.get('Description', '').strip()

        # If no description, automatically keep the company - cannot eliminate without information
        if not description:
            return {
                "should_keep": True,
                "keep": "Yes",
                "reason": "No description provided - keeping for detailed validation",
                "company_name": company_name,
                "filter_stage": "preliminary"
            }

        context_parts.append(f"Description: {description}")

        if company_data.get('Keywords'):
            context_parts.append(f"Keywords: {company_data.get('Keywords')}")

        context = "\n".join(context_parts)

        # Simple filtering prompt - emphasize using description
        user_prompt = f"""Assess if this company should be eliminated from software company screening:

{context}

⚠️ ELIMINATE if the description shows the company IS one of these business types:

1. **Marketing/advertising/PR agency** - Focus on marketing/brand/creative services
2. **Consulting firm** - Management, IT, business, or digital consulting
3. **System integrator** - Implementing SAP/Oracle/Microsoft/Salesforce for clients
4. **Engineering consultancy** - MEP, maritime, energy, defense engineering services
5. **Manufacturer** - Making industrial equipment, hardware, electronics
6. **Testing/QA services** - Project-based software testing/quality engineering
7. **Distributor/reseller** - Selling/distributing hardware or software products
8. **BPO/call center** - Outsourcing, contact center, customer service operations
9. **Logistics/transportation** - Freight, shipping, supply chain operations
10. **Training/coaching** - Leadership development, training services, HR consulting
11. **Construction/facilities** - Building installations, facilities management
12. **Traditional services** - Insurance, financial services (non-tech), publications, recruitment/staffing

**IMPORTANT LOGIC:**
- IF description clearly states they ARE one of the above business types → ELIMINATE
- ONLY KEEP if description explicitly mentions they HAVE proprietary software/SaaS products

**Examples:**
- "Management consulting firm" → ELIMINATE (even if description mentions "digital" or "tech")
- "IT consulting services" → ELIMINATE (even if they work with technology)
- "Recruitment agency" → ELIMINATE (it's staffing/HR services)
- "Hardware manufacturer" → ELIMINATE (even if products have some electronics)
- "System integrator for Microsoft" → ELIMINATE (implementing third-party software)

**KEEP if:**
- Description mentions proprietary software products, platforms, or SaaS
- Company is clearly a software vendor/provider
- You see terms like "our software", "our platform", "SaaS", "software solutions we built", "recurring revenue"

⚠️ If description says they ARE a consulting/services/manufacturing/etc company but doesn't mention software → ELIMINATE

---
**Output requirements**
• Respond **only** with a valid JSON object.
• Do **not** wrap it in backticks or code fences.
• Use exactly these keys:
  - "keep" ("Yes" or "No")
  - "reason" (one sentence: state the business type and whether they have proprietary software)

**Answer:**
- "No" (ELIMINATE) - If they ARE one of the 12 business types listed AND description doesn't mention proprietary software
- "Yes" (KEEP) - If description mentions proprietary software/SaaS OR business type is unclear"""

        try:
            # Combine system and user prompts
            combined_input = f"{self.system_prompt}\n\n{user_prompt}"

            # Use cheapest model WITHOUT web search for speed
            response = self.client.responses.create(
                model="gpt-5-nano",  # Cheapest model
                input=combined_input
                # NO web search tools - keep it fast and cheap
            )

            result = json.loads(response.output_text)

            should_keep = result.get('keep', '').lower() in ['yes', 'true']

            return {
                "should_keep": should_keep,
                "keep": result.get('keep', 'Yes'),
                "reason": result.get('reason', 'Unknown'),
                "company_name": company_name,
                "filter_stage": "preliminary"
            }

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse AI response: {str(e)}"
            return {
                "should_keep": True,
                "keep": "Yes",
                "reason": f"Error in preliminary filter: {error_msg}",
                "company_name": company_name,
                "filter_stage": "preliminary"
            }

        except Exception as e:
            error_msg = f"Error during filtering: {str(e)}"
            return {
                "should_keep": True,
                "keep": "Yes",
                "reason": f"Error in preliminary filter: {error_msg}",
                "company_name": company_name,
                "filter_stage": "preliminary"
            }

    def process_csv(self, input_path: str, output_path: str = None):
        """Process all companies from CSV and save filtered ones"""
        input_file = Path(input_path)
        if output_path is None:
            output_path = input_file.parent / "preliminary_filtered_companies.csv"

        kept_companies = []
        eliminated_companies = []
        filter_results = []

        # Read CSV and process each company
        with open(input_file, 'r', encoding='utf-8') as f:
            # Check if this is the original input.csv (has extra header rows)
            first_line = f.readline()
            f.seek(0)  # Reset to beginning

            if 'Search Url' in first_line:
                # Original input.csv - skip first two rows
                next(f)
                next(f)
            # else: deduplicated CSV - no need to skip rows

            reader = csv.DictReader(f)
            companies = list(reader)

        for i, row in enumerate(companies, 1):
            if row.get('Company Name'):
                result = self.filter_company(row)
                filter_results.append(result)

                if result.get('should_keep', True):
                    kept_companies.append(row)
                else:
                    eliminated_companies.append(row)

        # Save kept companies to new CSV
        if kept_companies:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=kept_companies[0].keys())
                writer.writeheader()
                writer.writerows(kept_companies)

        # Save filter results as JSON for reference
        results_path = Path(output_path).parent / "preliminary_filter_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(filter_results, f, indent=2)

        # Save eliminated companies separately for review
        if eliminated_companies:
            eliminated_path = Path(output_path).parent / "preliminary_eliminated.csv"
            with open(eliminated_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=eliminated_companies[0].keys())
                writer.writeheader()
                writer.writerows(eliminated_companies)

        return kept_companies, eliminated_companies, filter_results


def main():
    """Main execution function"""
    import sys

    filter_tool = PreliminaryFilter()

    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = "../0dedupe/deduplicated_companies.csv"

    output_csv = "preliminary_filtered_companies.csv"

    kept, eliminated, results = filter_tool.process_csv(input_csv, output_csv)

    print(f"Done — kept {len(kept)}, eliminated {len(eliminated)} of {len(results)}")
    print(f"Output: {output_csv}")


if __name__ == "__main__":
    main()
