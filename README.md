# Tree Species Classification

**Course**: Traitement des données multimédia  
**Features**: LBP (2D) + FPFH (3D) + RBF SVM  
**Dataset**: 691 point clouds, 7 tree species  

## Quick Start
```bash
python main.py                    # Show dataset overview
python part1_advanced.py          # Visualize point clouds
```

## Python Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `main.py` | 🎯 **Main entry point** | Start here! |
| `src/dataset_explorer.py` | 📊 **Data loading & stats** | Import for ML pipeline |
| `part1_advanced.py` | 🔍 **Visualization tool** | Explore individual point clouds |

##  Next Steps
1. Create `src/feature_extraction.py` (LBP + FPFH)
2. Create `src/classification.py` (RBF SVM)
3. Handle class imbalance in evaluation

## Dataset Info
- **Train**: 557 files (80.6%)
- **Test**: 134 files (19.4%) 
- **Imbalance**: Oak(22) vs Douglas Fir(183) - 8x difference!

---
*Clean, focused, ready for ML implementation* 🚀
