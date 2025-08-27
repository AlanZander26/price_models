# Contains the PriceModel class

from abc import ABC

#################################
# PriceModel Class
#################################

class PriceModel(ABC):
    """
    Represents the price model of a financial asset, such as a stock or derivative.

    Attributes
    ----------
    name : str
        Name of the price model (e.g., "StockPriceModel").
    """
    
    def __repr__(self) -> str:
        """
        Provide a string representation of the price model.

        Returns
        -------
        str
            A string in the format: ParentClass(CurrentClass).
        """
        # Get the immediate parent class (excluding object and self)
        parent_class = self.__class__.__bases__[0].__name__
        # Get the current class
        current_class = self.__class__.__name__
        # Format as ParentClass(CurrentClass)
        return f"{parent_class}({current_class})"
       