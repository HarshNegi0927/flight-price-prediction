import pickle
import xgboost as xgb

# pickle se load karo
with open("xgboost-model", "rb") as f:
    model = pickle.load(f)

# JSON mein save karo
model.save_model("xgboost-model.json")
print("Done!")