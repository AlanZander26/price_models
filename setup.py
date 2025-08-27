from setuptools import setup, find_packages

setup(
    name="price_models",
    version="0.1.0",
    description="A Python framework for modeling stock behavior and pricing options, including analytical models, risk sensitivities, probability analysis, and Monte Carlo simulations.",
    author="AZ",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",   # if you want plotting in examples
    ],
    python_requires=">=3.8",
)
