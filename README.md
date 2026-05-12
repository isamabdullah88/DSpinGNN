# DSpinGNN: Deep Learning for Macroscopic Spin-Lattice Dynamics in 2D Magnetism

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **DSpinGNN**, a physics-informed, $E(3)$-equivariant Graph Neural Network developed for my MS Physics thesis. 

DSpinGNN bridges the severe length-scale gap between first-principles quantum mechanics and macroscopic molecular dynamics. By simultaneously predicting interatomic potentials and dynamic magnetic exchange couplings ($J$), this framework enables the unprecedented simulation of mesoscopic spin-lattice entanglement in strain-engineered 2D magnetic materials, with a primary focus on **Chromium Triiodide (CrI₃)**.

---

## Emergent Magnetic Phase Coexistence

*Watch the macroscopic spin-lattice dynamics simulation below:*

[![Watch the Simulation](https://img.youtube.com/vi/Ma-eBKL-Knc/maxresdefault.jpg)](https://youtu.be/Ma-eBKL-Knc)
*(Click the image to play the video).*

In this 3,200-atom simulation, a propagating in-plane acoustic strain wave organically induces the formation of **oscillatory, concentric domain walls**. The network successfully resolves a complex topological magnetic landscape, capturing the dynamic "breathing" cycle of transient Ferromagnetic (FM) and Antiferromagnetic (AFM) phases. 

---

## Key Scientific Breakthroughs

### 1. Massive Length-Scale Transferability (8 to 3,200 Atoms)
First-principles Density Functional Theory (DFT) is intrinsically restricted to a few hundred atoms due to its $O(N^3)$ scaling. DSpinGNN breaks this bottleneck. Trained strictly on computationally inexpensive 8-atom primitive unit cells, the network achieves remarkable structural equivariance. This allows it to be scaled up by a factor of 400 to drive a stable 3,200-atom simulation without unphysical error accumulation.

### 2. Physics-Informed Extrapolation
Rather than relying on black-box data fitting, DSpinGNN's $\Delta$-MLP exchange predictor explicitly embeds the quantum mechanical **Goodenough-Kanamori superexchange rules** alongside an orbital overlap decay function. Because of this strong analytical inductive bias, the model robustly extrapolates complex, non-linear phase transitions well beyond its original training manifold during extreme localized structural deformations (up to 115° bond angles).

---

## Core Architecture & Pipeline

* **Bifurcated Equivariant Architecture**: Utilizes an Equivariant Graph Neural Network (based on NequIP) to predict Total Energy and Atomic Forces while strictly preserving $E(3)$ symmetries. Simultaneously, an independent physics-informed $\Delta$-MLP predicts continuous Edge-level Heisenberg Exchange parameters ($J_{ij}$).
* **High-Throughput DFT Data Generation**: Fully automated Python framework utilizing ASE to systematically generate and relax Uniaxial, Biaxial, and Shear strain configurations across periodic boundary conditions.
* **First-Principles Spin Extraction**: Seamless integration of **Quantum ESPRESSO**, **Wannier90**, and **TB2J** to rigorously calculate highly localized magnetic exchange parameters via the magnetic force theorem.
* **Langevin Spin-Lattice Dynamics**: Coupling the DSpinGNN potential with ASE to drive large-scale, discrete time-step Langevin dynamics, mapping localized magnetic responses continuously across spatial and temporal dimensions.

---

## Physical Objectives & Future Applications

DSpinGNN establishes a highly scalable, computationally rigorous paradigm for uncovering complex spin-lattice entanglement. It is designed to solve critical bottlenecks in the commercialization of 2D spintronics:
1.  **Programmable Magnetic Landscapes**: Mapping the exact mechanical tipping points where propagating strain waves force localized magnetic phase transitions. 
2.  **Strain-Driven Device Engineering**: Providing a theoretical playground to design non-volatile, high-density, spin-based memory storage and logic devices driven by surface acoustic waves or substrate corrugations.

---

## Tech Stack

* **Deep Learning**: PyTorch, PyTorch Geometric (PyG), e3nn (Euclidean Neural Networks)
* **First-Principles Physics**: Quantum ESPRESSO, Wannier90, TB2J
* **Dynamics & Visualization**: ASE (Atomic Simulation Environment), OVITO, Matplotlib
* **Infrastructure**: LUMS High-Performance Computing Cluster, DigitalOcean GPU Droplets

---
**Author**: Isam Balghari  
**Institution**: Lahore University of Management Sciences (LUMS)  
**Degree**: MS Physics  
**Date**: May 2026
