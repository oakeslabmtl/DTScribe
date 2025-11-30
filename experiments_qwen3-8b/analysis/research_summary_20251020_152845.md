# Digital Twin Characteristics Extraction - Experiment Analysis Report
Generated on: 2025-10-20 15:28:45

## Characteristics Extraction Analysis
- **Total Experiments**: 72
- **Average Extraction Rate**: 89.55% ± 2.95%
- **Best Extraction Rate**: 95.24%
- **Average Processing Time**: 151.53s
- **Average Description Length**: 479 characters

### Hyperparameter Impact Analysis
#### Chunk Size Impact on Extraction Rate
                 mean       std  count
chunk_size                            
2000        89.880952  3.528032     24
2500        89.285714  3.217447     24
3000        89.484127  1.975482     24

#### Temperature Impact on Extraction Rate
                  mean       std  count
temperature                            
0.1          89.682540  2.688856     24
0.2          89.682540  2.688856     24
0.3          89.285714  3.510523     24

## Quality Trends Over Time
## Recommendations
### Best Performing Configuration
- **Model**: qwen3:8b
- **Chunk Size**: 2000
- **Chunk Overlap**: 400
- **Temperature**: 0.1
- **Extraction Rate**: 95.24%

### Factors most correlated with extraction rate:
- **chunk_overlap**: 0.101
- **chunk_size**: 0.055
- **temperature**: 0.055

### Error Analysis
- **Experiments with errors**: 0.0%
