"""Custom exceptions for the Titanic Survival Analysis project."""

class TitanicBaseException(Exception):
    """Base exception for all custom exceptions in the project."""
    pass

class ValidationError(TitanicBaseException):
    """Raised when data validation fails."""
    pass

class DataNotFoundError(TitanicBaseException):
    """Raised when required data is not found."""
    pass

class ModelNotFoundError(TitanicBaseException):
    """Raised when a required model is not found."""
    pass

class FeatureEngineeringError(TitanicBaseException):
    """Raised when feature engineering fails."""
    pass

class PredictionError(TitanicBaseException):
    """Raised when model prediction fails."""
    pass

class DatabaseError(TitanicBaseException):
    """Raised when database operations fail."""
    pass

class ConfigurationError(TitanicBaseException):
    """Raised when configuration is invalid or missing."""
    pass

class APIError(TitanicBaseException):
    """Raised when API operations fail."""
    pass

class AuthenticationError(TitanicBaseException):
    """Raised when authentication fails."""
    pass

class AuthorizationError(TitanicBaseException):
    """Raised when authorization fails."""
    pass
