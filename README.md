# 🚀 From Structure to Dynamics

[![Paper](https://img.shields.io/badge/Paper-Open_Access-blue)](link_to_paper)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official code for the paper:
> **"From Structure to Dynamics: A General View of Community-level Features"**  
> *Nicolàs Merino, Núria Galiana, Jean-François Arnoldi, Miguel B. Araújo*  
> Conference/Journal, Year

## 📌 Overview

This repository contains all the Julia code and data necessary to reproduce the analyses and figures presented in the article "From Structure to Dynamics: A General View of Community-level Features". It includes scripts for generating ecological communities, applying network modifications, computing community- and species-level metrics, and performing the statistical analyses described in the manuscript.

## 🗂️ Repository Structure

```bash
├── README.md               
├── Code/                   
├── Figures/                
├── Outputs/                # .jls objects to be saved
├── paper.pdf               # PDF of the paper
├── Project.toml            # Package dependencies
├── Manifest.toml           # Pinned package versions for exact reproducibility. 
├── LICENSE                 # License information
```

## ⚙️ Installation
To set up the environment and install all dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/FromStructureToDynamics.git
cd FromStructureToDynamics

# Start Julia with the project environment
julia --project=.

# Inside Julia:
using Pkg
Pkg.instantiate()
```



