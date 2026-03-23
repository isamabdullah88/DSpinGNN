# DSpinGNN: Disentangled Spin Graph Neural Network for 2D Magnetism

This repository contains the implementation of DSpinGNN, an $E(3)$-equivariant Graph Neural Network developed as the core research of my MS Physics thesis. Building upon a foundational implementation of DSpinGNN, this project extends equivariant architectures to simultaneously predict interatomic potentials and magnetic exchange interactions in strain-engineered 2D magnetic materials, with a primary focus on **Chromium Triiodide (CrI₃)**.


## 🚀 Project Overview

The goal of this project is to map the complex magneto-elastic tensor of 2D materials using a combination of High-Throughput Density Functional Theory (DFT) and Machine Learning. By predicting how structural deformations alter the Goodenough-Kanamori-Anderson (GKA) super-exchange pathways, DSpinGNN serves as a predictive engine for engineering Ferromagnetic (FM) to Antiferromagnetic (AFM) phase transitions.

### Key Features
* **Multi-Task Equivariant Architecture**: An extension of the DSpinGNN backbone that utilizes an advanced Edge Decoder to simultaneously predict Graph-level (Total Energy), Node-level (Atomic Forces), and Edge-level (Heisenberg Exchange $J_{ij}$) properties.
* **High-Throughput DFT Pipeline**: Fully automated Python framework utilizing ASE to systematically generate and relax Uniaxial, Biaxial, and Shear strain configurations across periodic boundary conditions.
* **First-Principles Spin Extraction**: Integration of **Quantum ESPRESSO**, **Wannier90**, and **TB2J** to rigorously extract magnetic exchange parameters from maximally localized Wannier functions.
* **Periodic Graph Construction**: Custom data loaders in PyTorch Geometric (PyG) that map distant unit-cell interactions using fractional lattice shift vectors ($R$), strictly preserving the physics of the infinite 2D crystal

---

## 📊 Physical Objectives & Applications

DSpinGNN is designed to solve critical bottlenecks in the commercialization of 2D spintronics:

1. **FM-AFM Phase Transitions**: Mapping the exact mechanical tipping points where strain forces a magnetic phase transition, enabling the design of Terahertz-speed, high-density, piezomagnetic memory devices.
2. **Curie Temperature ($T_C$) Engineering**: Identifying specific strain tensors that maximize orbital overlap and strengthen the Heisenberg exchange, providing a theoretical pathway to push 2D magnetism closer to room temperature.
3. **Physics-Informed Regularization**: By driving the network training with both Energy/Forces and tiny $J$ parameters (meV scale), the model learns a highly accurate internal representation of the chemical bonds before predicting the sensitive magnetic states.



---

## 🛠️ Tech Stack
* **Deep Learning**: PyTorch, PyTorch Geometric (PyG), e3nn (Euclidean Neural Networks)

* **First-Principles Physics**: Quantum ESPRESSO (DFT), Wannier90 (Disentanglement), TB2J (Magnetic Exchange)

* **Molecular Tools**: ASE (Atomic Simulation Environment)

* **Infrastructure**: High-Performance Computing (HPC) Clusters / DigitalOcean GPU Droplets

* **Data Serialization**: PyTorch Binary Datasets (.pt)

---

## 🔬 Thesis Context: From Molecules to Magnetism

This repository represents the advanced application phase of my MS Thesis. It directly builds upon my previous work verifying from-scratch $E(3)$-equivariant mechanics on standard organic molecular dynamics (e.g., Aspirin/MD17).

By upgrading the architecture to handle periodic boundary conditions, transition-metal $d$-orbitals, and edge-level tensor predictions, DSpinGNN bridges the gap between purely structural machine learning potentials and the predictive discovery of next-generation quantum magnetic materials.

---
**Author**: Isam Balghari  
**Degree**: MS Physics  
**Expected Completion**: May 2026
