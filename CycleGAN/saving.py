__all__ = 'load_model', 'save_model'

from keras import Model
from keras import model_from_json

def load_model(filename: str):
    """Loads the model architecture and weights as a usable Model.
    Requires '<filename>_model.json' and '<filename>_weights.h5'"""

    with open(filename + '_model.json', 'r') as fobj:
        model = model_from_json(fobj.read())

    model.load_weights(filename + '_weights.h5')
    return model

def save_model(model: Model):
    """Saves the model architecture and weights of given model into
    '<filename>_model.json' and '<filename>_weights.h5'"""

    with open(filename + '_model.json', 'w') as fobj:
        fobj.write(model.to_json())

    model.save_weights(filename + '_weights.h5')
