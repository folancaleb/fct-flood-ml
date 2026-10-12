# FCT Flood Susceptibility ML Project

This project organizes the notebook into a modular pipeline.

## Structure
- Part A: build base rasters, extract samples, train RF/XGBoost/CART, evaluate, explain, map susceptibility
- Part B: ward-level composite risk, exposure, and manuscript outputs

## Run
```bash
python run_part_a.py
python run_part_b.py
```

## Notes
- Set paths in `config/defaults.py` or override them in `config/settings.py`
- The notebook logic is still the source of truth; this structure simply refactors it into modules
