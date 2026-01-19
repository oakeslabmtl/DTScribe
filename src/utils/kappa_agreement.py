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

# Quick validation to ensure data is entered correctly
for row in data:
    assert sum(row) == 3, f"Row {row} does not sum to 3 raters!"

# --- STEP 2: CALCULATE KAPPA ---
kappa = fleiss_kappa(data, method='fleiss')

print(f"Fleiss' Kappa: {kappa:.3f}")