from setuptools import setup, find_packages

setup(
    name="price_models",
    version="0.1.0",
    description="A Python framework for modeling stock behavior and pricing options, including analytical models, risk sensitivities, probability analysis, and Monte Carlo simulations.",
    author="AZ",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26,<3",
        "scipy>=1.11,<2",
        "matplotlib>=3.7,<4", # If you want to plot in 'examples'
    ],
    python_requires=">=3.8",
)

