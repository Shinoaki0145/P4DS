# P4DS — Python for Data Science (Notebooks & Examples)

A small collection of teaching and demo materials for Python-based data
exploration and visualization. This repository contains Jupyter notebooks,
datasets, and example assignments used for learning and teaching core data
science tools (NumPy, pandas, Matplotlib, etc.).

**Quick Links**

- Repository root: `P4DS`
- Environment file: `min_ds-env.yml`
- Data folder: `Data/`

**Contents**

- `*.ipynb` — Jupyter notebooks with demos and assignments (see filenames in
	the repo root and `L0/`, `L1/` folders).
- `Data/` — sample CSV and text datasets used by the notebooks (e.g. `Iris.csv`,
	`Cars.csv`, `P4DS-Grades.csv`).
- `min_ds-env.yml` — conda environment specification for running the notebooks.
- `nbdime/` — included library sources and examples (local copy used by some
	exercises).

Getting started
---------------

1. Create the environment (requires conda or mamba):

```bash
# with conda
conda env create -f min_ds-env.yml -n p4ds
conda activate p4ds

# or with mamba
mamba env create -f min_ds-env.yml -n p4ds
mamba activate p4ds
```

2. Launch Jupyter Lab or Notebook from the repository root:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open any demo notebook (filenames start with numbers indicating the
	 lesson, e.g. `06-Numpy_Demo.ipynb`, `08-Pandas_Demo.ipynb`).

Notes
-----

- The environment file `min_ds-env.yml` pins common data-science packages used
	by the notebooks. If you prefer a lightweight setup, install only the
	packages you need (e.g. `numpy`, `pandas`, `matplotlib`, `jupyter`).
- Datasets live in the `Data/` folder. Feel free to add new data files and
	reference them by relative path from notebooks.

Contributing
------------

- Improve notebooks or add new exercises via pull requests.
- If you add dependencies, update `min_ds-env.yml` accordingly.

License
-------

No license file is included in this repository. Add a `LICENSE` file if you
want to make the project explicitly open-source.

Contact
-------

For questions or suggestions, open an issue or contact the repository owner.
