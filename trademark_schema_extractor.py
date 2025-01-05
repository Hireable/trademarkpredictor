from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel
import pinecone
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime

class TrademarkVectorSystem:
    """
    A system for storing and analyzing trademark decision data using vector embeddings.
    This system allows us to:
    1. Convert trademark cases into vector embeddings
    2. Store these vectors with their associated metadata
    3. Predict confusion likelihood based on similar historical cases
    4. Provide explanations for predictions based on relevant precedents
    """
    
    def __init__(self, 
                 project_id: str,
                 pinecone_api_key: str,
                 pinecone_environment: str,
                 index_name: str = "trademark-decisions",
                 location: str = "europe-west2"):
        """
        Initializes the vector database system with necessary services.
        Uses Application Default Credentials for Google Cloud authentication.
        
        Args:
            project_id: Google Cloud project identifier
            pinecone_api_key: API key for Pinecone vector database
            pinecone_environment: Pinecone environment (e.g., "us-west1-gcp")
            index_name: Name for the Pinecone index
            location: Google Cloud region
        """
        try:
            # Initialize Vertex AI using ADC
            aiplatform.init(
                project=project_id,
                location=location,
                # The following flags help with enterprise environments
                encryption_spec_key_name=None,  # Use default encryption
                experiment=None,
                staging_bucket=None
            )
            
            # Initialize the model
            self.model = GenerativeModel("gemini-pro")
            
            print("Successfully authenticated with Google Cloud")
        except Exception as e:
            print(f"Error initializing Google Cloud services: {str(e)}")
            print("Please ensure you have run 'gcloud auth application-default login'")
            raise
        
        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key, 
                     environment=pinecone_environment)
                     
        # Create or connect to the index
        if index_name not in pinecone.list_indexes():
            # Create a new index with appropriate dimensions for our embeddings
            pinecone.create_index(
                name=index_name,
                dimension=768,  # Adjust based on your embedding model
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
        
        # Define the features we'll use for similarity comparison
        self.comparison_features = [
            'visual_similarity',
            'aural_similarity',
            'conceptual_similarity',
            'goods_services_similarity',
            'market_overlap',
            'consumer_attention_level',
            'mark_distinctiveness'
        ]

    def store_case(self, case_data: Dict[str, Any]) -> bool:
        """
        Stores a trademark case in the vector database.
        
        Args:
            case_data: Structured case data including comparison features
            and outcome information
        
        Returns:
            Boolean indicating success
        """
        try:
            # Extract features for embedding
            features = self._extract_features(case_data)
            
            # Create a detailed text representation for embedding
            text_representation = self._create_text_representation(features)
            
            # Generate embedding using Vertex AI
            embedding = self._generate_embedding(text_representation)
            
            # Create metadata for storage
            metadata = {
                'case_ref': case_data.get('case_metadata', {}).get('case_ref'),
                'filing_date': case_data.get('case_metadata', {}).get('filing_date'),
                'marks': {
                    'mark1': case_data.get('marks', {}).get('contested_mark'),
                    'mark2': case_data.get('marks', {}).get('earlier_marks', [{}])[0].get('mark')
                },
                'outcome': {
                    'confusion_found': case_data.get('outcome', {}).get('confusion_found'),
                    'decision': case_data.get('outcome', {}).get('decision')
                },
                'features': features,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in Pinecone
            self.index.upsert(
                vectors=[(
                    metadata['case_ref'],  # Use case reference as ID
                    embedding.tolist(),
                    metadata
                )]
            )
            
            return True
            
        except Exception as e:
            print(f"Error storing case: {str(e)}")
            return False

    def predict_confusion(self, 
                         input_features: Dict[str, float],
                         num_similar_cases: int = 5) -> Dict[str, Any]:
        """
        Predicts likelihood of confusion based on input features and similar cases.
        
        Args:
            input_features: Dictionary of comparison features with normalized values
            num_similar_cases: Number of similar cases to consider
        
        Returns:
            Dictionary containing prediction and supporting information
        """
        try:
            # Create text representation of input features
            text_representation = self._create_text_representation(input_features)
            
            # Generate embedding
            query_embedding = self._generate_embedding(text_representation)
            
            # Find similar cases
            similar_cases = self.index.query(
                vector=query_embedding.tolist(),
                top_k=num_similar_cases,
                include_metadata=True
            )
            
            # Analyze similar cases to make prediction
            prediction_result = self._analyze_similar_cases(
                similar_cases,
                input_features
            )
            
            # Generate explanation using AI
            explanation = self._generate_prediction_explanation(
                prediction_result,
                similar_cases,
                input_features
            )
            
            return {
                'prediction': prediction_result['prediction'],
                'confidence': prediction_result['confidence'],
                'similar_cases': prediction_result['similar_cases'],
                'explanation': explanation
            }
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None

    def _extract_features(self, case_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extracts and normalizes comparison features from case data.
        """
        features = {}
        
        # Extract and normalize each feature
        comparison = case_data.get('comparison', {})
        features['visual_similarity'] = self._normalize_similarity(
            comparison.get('visual_similarity'))
        features['aural_similarity'] = self._normalize_similarity(
            comparison.get('aural_similarity'))
        features['conceptual_similarity'] = self._normalize_similarity(
            comparison.get('conceptual_similarity'))
            
        # Extract goods/services similarity
        goods_services = case_data.get('goods_services', {})
        features['goods_services_similarity'] = self._normalize_similarity(
            goods_services.get('similarity'))
            
        # Additional features
        features['market_overlap'] = self._normalize_similarity(
            goods_services.get('overlap'))
        features['consumer_attention_level'] = self._normalize_attention_level(
            case_data.get('consumer_factors', {}).get('attention'))
        features['mark_distinctiveness'] = self._normalize_similarity(
            case_data.get('distinctiveness', {}).get('inherent'))
            
        return features

    def _create_text_representation(self, features: Dict[str, float]) -> str:
        """
        Creates a textual representation of features for embedding.
        """
        text_parts = []
        
        for feature, value in features.items():
            # Convert normalized values to descriptive text
            description = self._value_to_description(feature, value)
            text_parts.append(f"{feature.replace('_', ' ').title()}: {description}")
            
        return " | ".join(text_parts)

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding vector for the input text using Vertex AI.
        """
        response = self.model.generate_content(
            f"Create an embedding representation of this trademark comparison:\n{text}"
        )
        # Note: This is a simplified representation. In practice, you would use
        # a specific embedding model or endpoint.
        return np.array(response.embedding)

    def _analyze_similar_cases(self, 
                             similar_cases: List[Dict[str, Any]],
                             input_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes similar cases to generate a prediction.
        """
        weighted_predictions = []
        similar_cases_info = []
        
        for case in similar_cases.matches:
            # Calculate similarity score based on feature weights
            similarity_score = self._calculate_similarity_score(
                input_features,
                case.metadata['features']
            )
            
            # Get the actual outcome
            confusion_found = case.metadata['outcome']['confusion_found']
            
            # Add to weighted predictions
            weighted_predictions.append((confusion_found, similarity_score))
            
            # Store case information for reference
            similar_cases_info.append({
                'case_ref': case.metadata['case_ref'],
                'similarity_score': similarity_score,
                'outcome': case.metadata['outcome'],
                'features': case.metadata['features']
            })
        
        # Calculate final prediction and confidence
        prediction, confidence = self._calculate_final_prediction(
            weighted_predictions)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'similar_cases': similar_cases_info
        }

    def _generate_prediction_explanation(self,
                                      prediction_result: Dict[str, Any],
                                      similar_cases: List[Dict[str, Any]],
                                      input_features: Dict[str, float]) -> str:
        """
        Generates a human-readable explanation of the prediction using AI.
        """
        # Create a detailed prompt for the AI
        prompt = f"""
        Explain this trademark confusion prediction:
        
        Input Features:
        {json.dumps(input_features, indent=2)}
        
        Prediction:
        - Likelihood of Confusion: {prediction_result['prediction']}
        - Confidence: {prediction_result['confidence']}
        
        Similar Cases:
        {json.dumps(prediction_result['similar_cases'], indent=2)}
        
        Provide a clear explanation of:
        1. The main factors influencing this prediction
        2. How the similar cases support this conclusion
        3. Any important considerations or caveats
        
        Format the response in clear, professional language suitable for trademark lawyers.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

    def _normalize_similarity(self, value: Any) -> float:
        """
        Normalizes similarity values to a 0-1 scale.
        """
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, value / 5.0))
        return 0.0

    def _normalize_attention_level(self, value: str) -> float:
        """
        Converts attention level descriptions to normalized values.
        """
        mapping = {
            'very low': 0.0,
            'low': 0.25,
            'average': 0.5,
            'high': 0.75,
            'very high': 1.0
        }
        return mapping.get(str(value).lower(), 0.5)

    def _value_to_description(self, feature: str, value: float) -> str:
        """
        Converts normalized values back to descriptive text.
        """
        if value >= 0.8:
            return "very high"
        elif value >= 0.6:
            return "high"
        elif value >= 0.4:
            return "medium"
        elif value >= 0.2:
            return "low"
        else:
            return "very low"

    def _calculate_similarity_score(self,
                                  features1: Dict[str, float],
                                  features2: Dict[str, float]) -> float:
        """
        Calculates a weighted similarity score between two sets of features.
        """
        weights = {
            'visual_similarity': 0.25,
            'aural_similarity': 0.20,
            'conceptual_similarity': 0.20,
            'goods_services_similarity': 0.15,
            'market_overlap': 0.10,
            'consumer_attention_level': 0.05,
            'mark_distinctiveness': 0.05
        }
        
        score = 0.0
        for feature, weight in weights.items():
            if feature in features1 and feature in features2:
                difference = abs(features1[feature] - features2[feature])
                score += (1 - difference) * weight
                
        return score

    def _calculate_final_prediction(self,
                                  weighted_predictions: List[Tuple[bool, float]]
                                  ) -> Tuple[float, float]:
        """
        Calculates final prediction and confidence from weighted predictions.
        """
        if not weighted_predictions:
            return 0.5, 0.0
            
        total_weight = sum(weight for _, weight in weighted_predictions)
        if total_weight == 0:
            return 0.5, 0.0
            
        weighted_sum = sum(
            prediction * weight 
            for prediction, weight in weighted_predictions
        )
        
        prediction = weighted_sum / total_weight
        
        # Calculate confidence based on consistency of similar cases
        variations = [
            abs(pred - prediction) * weight
            for pred, weight in weighted_predictions
        ]
        confidence = 1.0 - (sum(variations) / total_weight)
        
        return prediction, confidence

if __name__ == "__main__":
    # Example usage
    vector_system = TrademarkVectorSystem(
        project_id="trademark-case-agent",
        pinecone_api_key="your-api-key",
        pinecone_environment="your-environment"
    )
    
    # Example input features
    input_features = {
        'visual_similarity': 0.8,
        'aural_similarity': 0.7,
        'conceptual_similarity': 0.9,
        'goods_services_similarity': 0.6,
        'market_overlap': 0.5,
        'consumer_attention_level': 0.3,
        'mark_distinctiveness': 0.4
    }
    
    # Get prediction
    prediction = vector_system.predict_confusion(input_features)
    
    if prediction:
        print("\nPrediction Results:")
        print(f"Likelihood of Confusion: {prediction['prediction']:.2%}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print("\nExplanation:")
        print(prediction['explanation'])
    else:
        print("Failed to generate prediction")