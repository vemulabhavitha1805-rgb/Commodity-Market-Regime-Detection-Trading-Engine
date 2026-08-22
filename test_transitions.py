from src.data_loader import load_data
from src.features import create_features
from src.regime_detection import detect_regimes
from src.transition_analysis import transition_matrix


data = load_data()

df = create_features(data)

df, model, scaler = detect_regimes(df)

matrix = transition_matrix(df)


print("\nRegime Transition Matrix")
print("=" * 70)

print(matrix)