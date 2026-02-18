import pandas as pd
import altair as alt
import re
import os

# Load the data
# Adjust path to be robust relative to script or simple relative path
data_path = 'data/dt_char_presence.csv'
if not os.path.exists(data_path):
    # Try finding it relative to project root if running from src/viz
    data_path = '../../data/dt_char_presence.csv'

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: Could not find data file at {data_path}")
    exit(1)

# 1. Define characteristic columns (Original headers in CSV)
characteristic_cols = [
    'MC1: System-under-Study (SUS)', 'MC2: Physical acting components',
    'MC3: Physical sensing components', 'MC4: Physical-to-Virtual Interaction',
    'MC5: Virtual-to-Physical Interaction', 'MC6: Digital Twin Services',
    'MC7: Twinning Timescale', 'MC8: Multiplicities', 'MC9: Life-cycle stages',
    'MC10: Digital Twin Models and Data', 'MC11: Tooling and Enablers',
    'MC12: Digital Twin Constellation', 'MC13: Twinning Process and Digital Twin Evolution',
    'MC14: Fidelity and Validity Considerations', 'MC15: Digital Twin Technical Connection',
    'MC16: Digital Twin Hosting/Deployment', 'MC17: Insights and Decision Making',
    'MC18: Horizontal Integration', 'MC19: Data Ownership and Privacy',
    'MC20: Standardization', 'MC21: Security and Safety Considerations'
]

# New labels from user request
new_labels_text = [
    "System under study", "Physical acting components", "Physical sensing components",
    "Physical-to-virtual interaction", "Virtual-to-physical interaction", "DT services",
    "Twinning time-scale", "Multiplicities", "Life-cycle stages", "DT models and data",
    "Tooling and enablers", "DT constellation", "Twinning process and DT evolution",
    "Fidelity and validity considerations", "DT technical connection", "DT hosting/deployment",
    "Insights and decision making", "Horizontal integration", "Data ownership and privacy",
    "Standardization", "Security and safety considerations"
]

# Mapping from CSV header to "Cn: {Name}"
col_mapping = {}
for idx, old_col in enumerate(characteristic_cols):
    c_num = idx + 1
    col_mapping[old_col] = f"C{c_num}: {new_labels_text[idx]}"

paper_col = 'Which paper are you rating?'

# 2. Assign Numeric IDs to Papers
unique_papers = sorted(df[paper_col].unique())
paper_id_map = {title: i+1 for i, title in enumerate(unique_papers)}

print("="*60)
print("PAPER ID MAPPING (Use this for your caption/legend):")
print("-" * 60)
for title, pid in paper_id_map.items():
    print(f"P{pid}: {title}")
print("="*60)

df['Paper_ID_Num'] = df[paper_col].map(paper_id_map)
df['Paper_Label'] = df['Paper_ID_Num'].apply(lambda x: f"P{x}")

# 3. Melt the data
df_melted = df.melt(
    id_vars=[paper_col, 'Paper_Label', 'Paper_ID_Num'],
    value_vars=characteristic_cols,
    var_name='Characteristic',
    value_name='Rating'
)

# 4. Calculate consensus
# 'Present' -> 1, 'Absent' -> 0
df_melted['Present_Value'] = df_melted['Rating'].apply(lambda x: 1 if x == 'Present' else 0)

# Group by paper and characteristic
df_heatmap = df_melted.groupby([
    'Paper_Label', 'Paper_ID_Num', paper_col, 'Characteristic'
]).agg(
    Present_Count=('Present_Value', 'sum')
).reset_index()

# 5. Sorting and Labels
# Extract the number from the original column name for sorting (MC1 -> 1)
df_heatmap['C_Number'] = df_heatmap['Characteristic'].str.extract(r'MC(\d+)').astype(int)

# Apply the new display labels using the mapping
df_heatmap['Display_Characteristic'] = df_heatmap['Characteristic'].map(col_mapping)

# 6. Map count to Descriptive Label for Legend
count_map = {
    0: 'Absent (0/3 votes)',
    1: 'Weak Presence (1/3 votes)',
    2: 'Moderate Presence (2/3 votes)',
    3: 'Strong Presence (3/3 votes)'
}
df_heatmap['Consensus_Label'] = df_heatmap['Present_Count'].map(count_map)

# 7. Colors: Sequential Blue Scale
domain = ['Absent (0/3 votes)', 'Weak Presence (1/3 votes)', 'Moderate Presence (2/3 votes)', 'Strong Presence (3/3 votes)']
range_colors = ['#f7f7f7', '#deebf7', '#9ecae1', '#08519c']

# 8. Chart Construction
base = alt.Chart(df_heatmap, title=alt.TitleParams(
    text="Digital Twin Characteristic Presence Consensus",
    fontSize=16,
    subtitle=["Aggregation of 3 Raters"],
    subtitleFontSize=12,
    anchor='start'
)).encode(
    x=alt.X('Display_Characteristic:N', 
            title=None, 
            sort=alt.EncodingSortField(field='C_Number', order='ascending'), 
            # Rotated labels (-45) to allow full names to be readable
            axis=alt.Axis(labelAngle=-45, labelFontSize=10, labelLimit=400)),
    y=alt.Y('Paper_Label:N', 
            title='Paper ID',
            # Sort numerically by the ID number P1, P2...
            sort=alt.EncodingSortField(field='Paper_ID_Num', order='ascending'),
            axis=alt.Axis(labelFontSize=11, labelFontWeight='bold')) 
)

heatmap = base.mark_rect(stroke='white', strokeWidth=1).encode(
    color=alt.Color('Consensus_Label:N',
                    scale=alt.Scale(domain=domain, range=range_colors),
                    legend=alt.Legend(
                        title="Consensus Level",
                        orient='bottom',
                        titleFontSize=12,
                        labelFontSize=11,
                        direction='horizontal'
                    )
                   ),
    tooltip=[
        alt.Tooltip(paper_col, title='Full Paper Title'),
        alt.Tooltip('Paper_Label:N', title='Paper ID'),
        alt.Tooltip('Display_Characteristic:N', title='Characteristic'),
        alt.Tooltip('Present_Count:Q', title='Votes'),
        alt.Tooltip('Consensus_Label:N', title='Status')
    ]
)

# Optional: Add text numbers for monochrome clarity
text = base.mark_text(baseline='middle').encode(
    text=alt.Text('Present_Count:Q'),
    color=alt.condition(
        alt.datum.Present_Count > 1,
        alt.value('white'),
        alt.value('black')
    )
)

final_chart = (heatmap + text).properties(
    width=alt.Step(40),
    height=alt.Step(40)
).configure_view(
    strokeWidth=0
).configure_axis(
    domain=False,
    ticks=False
)

alt.renderers.enable("html")
final_chart.save('assessment_heatmap.html')
print("Chart saved to assessment_heatmap.html")
