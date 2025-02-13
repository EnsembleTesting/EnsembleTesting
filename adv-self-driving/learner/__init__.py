from collections import defaultdict

from learner.driving_models_dave2v2 import Dave2v2
#from learner.driving_models_dave2v9 import Dave2v9
from learner.driving_models_dave2 import Dave2v1

_model_scope_dict = {
    'Dave2V2': Dave2v2,
    #'Dave2V9': Dave2v9,
    'Dave2V1': Dave2v1
}

model_scope_dict = defaultdict(lambda: None, **_model_scope_dict)
