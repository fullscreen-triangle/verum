"""
Verum Learning Engine

Personal Intelligence-Driven Navigation - Machine Learning Components

This package provides the machine learning and data processing capabilities
for the Verum autonomous driving framework, including:

- Cross-domain pattern learning (driving, walking, tennis, etc.)
- Personal AI model training and adaptation
- Biometric data analysis and pattern recognition
- Multi-domain data integration and preprocessing

Key Components:
- core: Core learning algorithms and pattern transfer
- data: Data collection, preprocessing, and integration
- models: Machine learning models and neural networks
- analysis: Analysis tools and visualization
"""

__version__ = "0.1.0"
__author__ = "Verum Team"
__email__ = "team@verum.ai"

# Core imports
from .core import (
    CrossDomainLearner,
    PatternTransfer,
    PersonalModelTrainer,
    ValidationFramework,
)

from .data import (
    DataCollector,
    DataPreprocessor,
    MultiDomainIntegrator,
    PrivacyPreserver,
)

from .models import (
    PersonalAIModel,
    CrossDomainNetwork,
    FearResponseModel,
    BiometricAnalyzer,
)

from .analysis import (
    PerformanceAnalyzer,
    PatternVisualizer,
    ModelValidator,
    ResearchTools,
)

# Configuration and utilities
from .config import LearningConfig, TrainingConfig, DataConfig
from .utils import (
    setup_logging,
    create_experiment,
    load_personal_data,
    export_model,
)

__all__ = [
    # Core learning components
    "CrossDomainLearner",
    "PatternTransfer", 
    "PersonalModelTrainer",
    "ValidationFramework",
    
    # Data processing components
    "DataCollector",
    "DataPreprocessor",
    "MultiDomainIntegrator",
    "PrivacyPreserver",
    
    # Model components
    "PersonalAIModel",
    "CrossDomainNetwork",
    "FearResponseModel",
    "BiometricAnalyzer",
    
    # Analysis components
    "PerformanceAnalyzer",
    "PatternVisualizer",
    "ModelValidator",
    "ResearchTools",
    
    # Configuration
    "LearningConfig",
    "TrainingConfig", 
    "DataConfig",
    
    # Utilities
    "setup_logging",
    "create_experiment",
    "load_personal_data",
    "export_model",
]

# Package metadata
DOMAINS_SUPPORTED = [
    "driving",
    "walking", 
    "cycling",
    "tennis",
    "basketball",
    "gaming",
    "daily_navigation",
]

BIOMETRIC_SENSORS = [
    "heart_rate",
    "skin_conductance", 
    "muscle_tension",
    "eye_tracking",
    "accelerometer",
    "gyroscope",
    "breathing",
    "temperature",
]

# Version information
def get_version_info():
    """Get detailed version information."""
    import sys
    import torch
    import numpy as np
    import sklearn
    
    return {
        "verum_learn": __version__,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }

def check_dependencies():
    """Check if all required dependencies are available."""
    required_packages = [
        "torch",
        "tensorflow", 
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "matplotlib",
        "seaborn",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install with: pip install {' '.join(missing_packages)}"
        )
    
    return True

# Initialize logging on import
def _setup_default_logging():
    """Setup default logging configuration."""
    import logging
    
    # Create logger
    logger = logging.getLogger("verum_learn")
    logger.setLevel(logging.INFO)
    
    # Create handler if none exists
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# Setup logging
logger = _setup_default_logging()
logger.info(f"Verum Learning Engine v{__version__} initialized")

# Check dependencies on import (optional, can be disabled)
import os
if os.getenv("VERUM_CHECK_DEPS", "true").lower() == "true":
    try:
        check_dependencies()
        logger.info("All dependencies verified")
    except ImportError as e:
        logger.warning(f"Dependency check failed: {e}") 