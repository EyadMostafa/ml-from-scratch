from abc import ABC, abstractmethod
from utils.helpers import validate_transform_input

class BaseModel(ABC):
    def __init__(self):
        pass

    _validate_transform_input = staticmethod(validate_transform_input)

    @property
    def get_params(self):
        """
        Get public, protected, and private parameters of the model.
    
        Returns
        -------
        params : dict
            A dictionary containing public, protected, and private attributes.
        """
        params = {}
    
        for attr, value in self.__dict__.items():
            if attr.startswith(f"_{self.__class__.__name__}__"):
                # Private attribute, remove class name prefix
                clean_name = attr.split(f"_{self.__class__.__name__}__")[-1]
            elif attr.startswith('_') and not attr.startswith('__'):
                # Protected attribute
                clean_name = attr[1:]
            elif not attr.startswith('_'):
                # Public attribute
                clean_name = attr
            else:
                continue  # Skip other double underscore built-ins
            
            params[clean_name] = value
    
        return params
    
        
    
    def set_params(self, **params):
        """
        Set public and protected parameters of the model.

        Parameters
        ----------
        **params : dict
            Dictionary of parameter names and values to set.

        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            # Try public attribute
            if hasattr(self, key):
                setattr(self, key, value)
            # Try protected attribute
            elif hasattr(self, f'_{key}'):
                setattr(self, f'_{key}', value)
            else:
                raise ValueError(f"Parameter '{key}' is not valid for {self.__class__.__name__}.")
        return self

    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass