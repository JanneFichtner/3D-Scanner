import pickle

with open(r"/home/janne/Desktop/Masterprojekt/calibration_results_thinlaser.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data)
