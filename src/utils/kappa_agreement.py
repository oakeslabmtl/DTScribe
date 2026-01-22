import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Data Format: [Count of 'Absent', Count of 'Present']
# Total subjects = 105 (5 papers * 21 characteristics)

# Example Results
# [3, 0] = Everyone agreed it was Absent
# [0, 3] = Everyone agreed it was Present
# [1, 2] = 1 said Absent, 2 said Present (Disagreement)

data = [
    # --- P1 (Row 1) ---
    [0, 3], [3, 0], [0, 3], [0, 3], [3, 0], [0, 3], [0, 3], [1, 2], [0, 3], [0, 3], [0, 3], [0, 3], [1, 2], [1, 2], [0, 3], [2, 1], [0, 3], [2, 1], [3, 0], [0, 3], [3, 0],
    
    # --- P2 (Row 2) ---
    [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [1, 2], [0, 3], [0, 3], [0, 3], [0, 3], [1, 2], [1, 2], [0, 3], [1, 2], [0, 3], [3, 0], [3, 0], [0, 3], [1, 2],

    # --- P3 (Row 3) ---
    [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [2, 1], [1, 2], [0, 3], [0, 3], [0, 3], [0, 3], [2, 1], [0, 3], [0, 3], [1, 2], [0, 3], [3, 0], [3, 0], [2, 1], [3, 0],

    # --- P4 (Row 4) ---
    [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [1, 2], [3, 0], [3, 0], [0, 3], [2, 1],

    # --- P5 (Row 5) ---
    [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [0, 3], [1, 2], [2, 1], [1, 2], [0, 3], [3, 0], [2, 1], [0, 3], [0, 3], [1, 2], [0, 3], [3, 0], [3, 0], [3, 0], [3, 0]
]

def classify_data(data_chunk):
    """
    Classifies a list of data points into a binary vector based on the rule:
    - [0, 3] or [1, 2] -> 1
    - Else -> 0
    """
    binary_vector = []
    for pair in data_chunk:
        if pair == [0, 3] or pair == [1, 2]:
            binary_vector.append(1)
        else:
            binary_vector.append(0)
    return binary_vector

# Quick validation to ensure data is entered correctly
for row in data:
    assert sum(row) == 3, f"Row {row} does not sum to 3 raters!"

# --- STEP 2: CALCULATE KAPPA ---
kappa = fleiss_kappa(data, method='fleiss')

print(f"Fleiss' Kappa: {kappa:.3f}")

# --- STEP 3: CLASSIFY DATA INTO BINARY VECTORS ---
vectors = {}

for i in range(5):
    # Slice the data for the current row
    start_index = i * 21
    end_index = start_index + 21
    current_chunk = data[start_index:end_index]
    
    # Classify the chunk
    vector_name = f"P{i+1}"
    vectors[vector_name] = classify_data(current_chunk)

for name, vec in vectors.items():
    print(f"{name}: {vec}")