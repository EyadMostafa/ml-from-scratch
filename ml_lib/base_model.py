from abc import ABC, abstractmethod
from utils.helpers import validate_transform_input

class BaseModel(ABC):
    def __init__(self):
        pass

    _validate_transform_input = staticmethod(validate_transform_input)
        
    def get_params(self):
        """
        Get parameters of the model.
    
        Returns
        -------
        params : dict
            A dictionary containing model parameters. This includes public, protected (_),
            and private (__) attributes with cleaned-up keys, removing leading underscores
            or class name prefixes for private variables.
        """
        cls_name = self.__class__.__name__
        private_prefix = f"_{cls_name}__"
        params = {}
    
        for attr, value in self.__dict__.items():
            if attr.startswith(private_prefix):
                clean_name = attr[len(private_prefix):]
            elif not attr.startswith('__'):
                clean_name = attr.lstrip('_')
            else:
                continue
    
            params[clean_name] = value
    
        return params
    
    
    def set_params(self, **params):
        """
        Set parameters of the model. Accepts public, protected (_), and private (__) attributes.
    
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
            # Try private attribute
            elif hasattr(self, f'_{self.__class__.__name__}__{key}'):
                setattr(self, f'_{self.__class__.__name__}__{key}', value)
            else:
                raise ValueError(f"Parameter '{key}' is not valid for {self.__class__.__name__}.")
        return self

    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass