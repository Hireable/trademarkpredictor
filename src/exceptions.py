class TrademarkAgentError(Exception):
    """Base exception class for TrademarkCaseAgent errors"""
    pass

class ConfigurationError(TrademarkAgentError):
    """Raised when there is an error in configuration settings"""
    pass

class ProcessingError(TrademarkAgentError):
    """Raised when there is an error processing a document"""
    pass

class ValidationError(TrademarkAgentError):
    """Raised when there is an error validating data"""
    pass

class ResourceError(TrademarkAgentError):
    """Raised when there is an error with external resources"""
    pass