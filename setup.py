"""
Industry Evaluation System Setup
"""

from setuptools import setup, find_packages

setup(
    name="industry-evaluation",
    version="0.1.0",
    description="Industry Model Evaluation System",
    packages=find_packages(include=['industry_evaluation*', 'src*']),
    python_requires=">=3.8",
    install_requires=[
        "pyyaml>=6.0.0",
        "requests>=2.31.0",
        "flask>=2.3.0",
        "flask-restx>=1.1.0",
        "flask-cors>=4.0.0",
        "watchdog>=3.0.0",
        "psutil>=5.9.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ]
    }
)