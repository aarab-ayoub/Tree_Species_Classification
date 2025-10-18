# Tree Species Classification

**Course**: Traitement des données multimédia  
**Features**: LBP (2D) + FPFH (3D) + RBF SVM  
**Dataset**: 691 point clouds, 7 tree species  

## Quick Start
```bash
python main.py                              # Basic dataset info
jupyter notebook notebooks/data_visualization.ipynb  # Data exploration
```

## Project Progress

### Phase 1: Data Foundation ✓ COMPLETE
- [x] Dataset download and organization
- [x] Train/test split (per teacher's test.csv)
- [x] Basic data loading functions

### Phase 2: Data Visualization ← CURRENT PHASE
- [x] Jupyter notebook for exploration
- [ ] Understand data characteristics
- [ ] Identify visualization patterns

### Phase 3: Feature Extraction (NEXT)
- [ ] LBP (Local Binary Patterns) - 2D descriptors
- [ ] FPFH (Fast Point Feature Histograms) - 3D descriptors

### Phase 4: Classification (FINAL)
- [ ] RBF SVM implementation
- [ ] Handle class imbalance (Oak: 22 vs Douglas Fir: 183)
- [ ] F1-score evaluation

## Files Structure

| File | Purpose |
|------|---------|
| `main.py` | Simple entry point |
| `src/simple_loader.py` | Basic point cloud loading |
| `notebooks/data_visualization.ipynb` | Data exploration |

## Dataset Info
- **Train**: 557 files (80.6%)
- **Test**: 134 files (19.4%)
- **Class Imbalance**: Largest class 8x bigger than smallest

---
*Focus: Simple, clean, step-by-step implementation*
