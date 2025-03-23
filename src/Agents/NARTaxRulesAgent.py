from typing import Any, Dict

from crewai import Agent


class TaxRulesAgent(Agent):
    tax_data: Dict[str, Any] = {}

    def __init__(self):
        super().__init__(
            name="Tax Rules Agent",
            role="Tax Compliance and Optimization Specialist",
            goal="Apply jurisdiction-specific tax rules and identify tax optimization strategies.",
            backstory=(
                "A financial AI expert in tax regulations, ensuring compliance while "
                "maximizing tax efficiency for portfolios."
            )
        )

    def apply_tax_rules(self, portfolio_data: Dict[str, Any], jurisdiction: str) -> Dict[str, Any]:
        """
        Apply jurisdiction-specific tax rules on portfolio data.
        """
        tax_rates = {
            "US": 0.15,
            "UK": 0.20,
            "EU": 0.18,
            "IN": 0.10
        }
        tax_rate = tax_rates.get(jurisdiction, 0.15)  # Default to US rate

        tax_liability = 0
        for holding in portfolio_data.get("holdings", []):
            current_price = holding.get("current_price", 200)  # Use real-time price or default to 200
            capital_gain = (holding["quantity"] * current_price) - (holding["quantity"] * holding["purchase_price"])
            tax = capital_gain * tax_rate
            tax_liability += tax
        
        self.tax_data = {
            "jurisdiction": jurisdiction,
            "tax_rate": tax_rate,
            "tax_liability": tax_liability
        }
        return self.tax_data

    def identify_tax_optimizations(self) -> Dict[str, Any]:
        """
        Identify opportunities for tax optimizations and offsets.
        """
        optimizations = []

        if self.tax_data.get("tax_liability", 0) > 1000:
            optimizations.append("Consider long-term holdings for reduced tax rates.")
        
        if self.tax_data.get("jurisdiction") == "US":
            optimizations.append("Utilize tax-loss harvesting to offset gains.")
        
        return {"optimizations": optimizations}

    def generate_tax_report(self) -> Dict[str, Any]:
        """
        Generate detailed tax liability reports for users.
        """
        report = {
            "Jurisdiction": self.tax_data.get("jurisdiction", "Unknown"),
            "Tax Rate": f"{self.tax_data.get('tax_rate', 0) * 100}%",
            "Total Tax Liability": self.tax_data.get("tax_liability", 0),
            "Optimizations": self.identify_tax_optimizations()["optimizations"]
        }
        return report

    def execute(self, portfolio_data: Dict[str, Any], jurisdiction: str) -> Dict[str, Any]:
        """
        Main method to execute tax rules, optimizations, and report generation.
        """
        self.apply_tax_rules(portfolio_data, jurisdiction)
        tax_report = self.generate_tax_report()
        return {
            "status": "success",
            "task": "tax_analysis",
            "tax_report": tax_report
        }
