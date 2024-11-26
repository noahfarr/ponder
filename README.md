# Ponder

Ponder is a repository dedicated to the implementation of model-based reinforcement learning (RL) algorithms in a simple, CleanRL-style single-file format. This repository is designed to be accessible, modular, and educational, making it an excellent resource for researchers, practitioners, and enthusiasts in the field of reinforcement learning.

---

## Features

- **Single-file Implementations**: Each algorithm is implemented in a self-contained Python script to ensure simplicity and readability.
- **Model-Based Focus**: The repository exclusively focuses on model-based RL, providing a diverse range of algorithms and their practical applications.
- **CleanRL-inspired Design**: Code is structured for clarity and ease of understanding, following the philosophy of the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository.
- **Reproducible Results**: Each implementation includes training scripts, hyperparameter settings, and evaluation tools to facilitate reproducibility.

---

## Algorithms Implemented

- **Dyna-Q**
- **MVE-DDPG** (WIP)

More algorithms are being actively developed and added to the repository.

I currently plan to implement:

- **World Models** (Inspired by the seminal paper by Ha and Schmidhuber, 2018)
- **PlaNet** (Planning with Latent Dynamics)
- **Dreamer**
- **MBPO** (Model-Based Policy Optimization)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/noahfarr/ponder.git
cd ponder
```

Install the dependencies:

```bash
poetry install
poetry shell
```

## Acknowledgments
Special thanks to the CleanRL project for inspiring the design philosophy of this repository and to the RL research community for their foundational contributions to model-based RL.
