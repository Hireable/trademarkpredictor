import os
from trademark_vector_system import TrademarkVectorSystem
from config import get_config

def main():
    """
    Main function to initialize and test the TrademarkVectorSystem
    """
    try:
        # Set environment variable for security (in production, set this in your environment)
        os.environ["PINECONE_API_KEY"] = "pcsk_pZVKP_LE6hxLQJpvXunANZNV4xUcB1mAA6p6SJmLFc1nNDTwssSH7zMSeW1bCz1XN4eX1"  # Replace with your actual API key
        
        # Get configuration
        config = get_config()
        
        # Initialize the system
        system = TrademarkVectorSystem(
            api_key=os.environ.get("PINECONE_API_KEY"),
        )

        # Sample case data for testing
        sample_case_data = {
            "case_metadata": {
                "case_ref": "TM12345",
                "officer": "John Smith",
                "dec_date": "2023-10-26",
                "jurisdiction": "UK"
            },
            "party_info": {
                "app_mark": "BrandX",
                "opp_mark": "BrandY",
                "app_name": "Applicant Inc.",
                "opp_name": "Opponent Ltd.",
                "market_presence": {
                    "opp_market_tenure": 5,
                    "geographic_overlap": 4
                }
            },
            "commercial_context": {
                "app_spec": "Electronic goods",
                "opp_spec": "Electronic devices",
                "app_class": [9],
                "opp_class": [9],
                "market_characteristics": {
                    "price_point": 3,
                    "purchase_frequency": 4,
                    "market_sophistication": 3
                }
            },
            "similarity_assessment": {
                "mark_similarity": {
                    "vis_sim": 4,
                    "aur_sim": 3,
                    "con_sim": 5
                },
                "gds_sim": {
                    "nature": 4,
                    "purpose": 5,
                    "channels": 3,
                    "use": 4
                }
            },
            "outcome": {
                "confusion": True,
                "conf_type": "indirect",
                "confidence_score": 4
            }
        }

        # Test the system by storing the sample case
        if system.upload_and_process_case(sample_case_data):
            print("Sample case stored successfully.")
        else:
            print("Failed to store sample case.")

    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()