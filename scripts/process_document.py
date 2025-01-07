from trademarkpredictor.agent import TrademarkCaseAgent
from pathlib import Path

def process_test_document(pdf_path):
    agent = TrademarkCaseAgent()
    
    # Example case data structure
    case_data = {
  "case_metadata": {
    "case_ref": "O/0703/24",
    "officer": "Sarah Wallace",
    "dec_date": "2024-07-25",
    "jurisdiction": "UK"
  },
  "party_info": {
    "app_mark": "JOLLY PECKISH",
    "opp_mark": "JOLLY",
    "app_name": "Stonegate Farmers Limited",
    "opp_name": "The Jolly Hog Group Limited",
    "market_presence": {
      "opp_market_tenure": 4,
      "geographic_overlap": null
    }
  },
  "commercial_context": {
    "app_spec": "Eggs; Birds egg products; Dairy products for food; Prepared meals; Snacks; Sandwiches; Meat pies; Cereal-based savoury snacks; Pastries, cakes, tarts, biscuits, breads.",
    "opp_spec": "Meat; Fish; Poultry; Prepared meals; Meat-based snacks; Sausage rolls; Vegan products; Catering services.",
    "app_class": [29, 30],
    "opp_class": [29, 30, 43],
    "market_characteristics": {
      "price_point": 3,
      "purchase_frequency": 4,
      "market_sophistication": 3
    }
  },
  "similarity_assessment": {
    "mark_similarity": {
      "vis_sim": 3,
      "aur_sim": 3,
      "con_sim": 3
    },
    "gds_sim": {
      "nature": 4,
      "purpose": 4,
      "channels": 3,
      "use": 3
    }
  },
  "market_dynamics": {
    "comp_gds": true,
    "comp_mkt": true,
    "actual_confusion": false,
    "market_overlap_duration": null
  },
  "consumer_factors": {
    "attention": 3,
    "purchase_environment": "both",
    "impulse_purchase_likelihood": 3
  },
  "distinctiveness": {
    "inherent": 3,
    "acquired": null,
    "distinct": 3
  },
  "outcome": {
    "confusion": true,
    "conf_type": "indirect",
    "confidence_score": 4
  }
}
    
    result = agent.process_case(case_data)
    print(f"Processing result: {result}")

if __name__ == "__main__":
    pdf_path = Path("\data\raw-data\raw_pdfs\o-0060-24 pet clothing v clothing.pdf"")
    process_test_document(pdf_path)