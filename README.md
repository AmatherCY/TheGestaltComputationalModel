# Code of "The Gestalt Computational Model by Persistent Homology"

## Overview

This project implements computational models for Gestalt visual perception principles including Proximity, Similarity, Closure, Good continuation and Pragnanz using topological data analysis (persistent homology).

## Project Structure

```
├── data/           # Input datasets
├── pic/            # Output visualizations
├── *.py           # Core implementation files
└── search_funcs.py # Utility functions
```

---

*This project bridges abstract visual perception theories with concrete computational models using topological data analysis.*

## Requirements

Install the required dependencies:

```bash
pip install gudhi
pip install ripser
pip install networkx
pip install scikit-learn
pip install matplotlib
pip install numpy scipy
```

## Running the Code

### Core Gestalt Principles

#### 1. Similarity Principle
```bash
python similarity1.py
python similarity2.py
```
- Groups elements based on visual similarity
- Creates similarity-based clustering visualizations
  
#### 2. Proximity Principle
```bash
python proximity.py
```
- Demonstrates grouping based on spatial proximity
- Generates proximity analysis visualization

#### 3. Closure Principle
```bash
python closure.py
```
- Detects and completes incomplete visual structures
- Uses data from `data/triangles.txt`
- Outputs closure detection results

#### 4. Good Continuation
```bash
python good_con.py
```
- Demonstrates the principle of good continuation
- Uses data from `data/goodcon.txt`

#### 5. Pragnanz Principle
```bash
python pragnanz.py
```
- Finds the simplest and most regular visual organization
- Uses data from `data/Olympics.txt`


### Results of Real images

Run these scripts to see results of real images in our paper:

```bash
python 4Rings_image.py

python Go_image.py

python UM_image.py

python candle_image.py

python waves_image.py

python wires_image.py
```