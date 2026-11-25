import numpy as np
import pandas as pd
import sklearn
import mlflow

frame = pd.DataFrame({"value": np.arange(3)})
assert frame["value"].sum() == 3
print("Environnement ML operationnel.")
