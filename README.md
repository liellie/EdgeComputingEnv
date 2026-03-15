# EdgeComputingEnv
EdgeComputingEnv: A discrete-event simulator for edge computing task scheduling. It models device heterogeneity, network topologies, and complex trust/security mechanisms. Designed for reproducible research in trust-aware scheduling, GNNs, and multi-objective optimization.

# EdgeComputingEnv Simulator

## Overview

**EdgeComputingEnv** is a discrete-event simulator developed for research in edge computing task scheduling. It is designed to faithfully reproduce complex edge scenarios, including device heterogeneity, network topologies, dynamic trust relationships, and security threats.

This simulator was used to evaluate the **GraphMatch** framework in the paper *"GraphMatch: A Graph-Based Trust-Aware Framework for Secure Multi-Objective Task Scheduling in Heterogeneous Edge Computing"* submitted to *Future Generation Computer Systems*.

## Hierarchical Architecture

The simulator follows a four-layer design:

1. **Configuration Layer**: Handles global parameters (n_devices, malicious_ratio, etc.)
2. **Entity Generation Layer**: Instantiates heterogeneous `EdgeDevice` and `Task` objects
3. **Relationship Modeling Layer**: Constructs small-world network topologies and pairwise trust matrices
4. **Interface Layer**: Provides standardized APIs for querying state and evaluating performance metrics

## Key Features

- **Trust Modeling**: Supports direct interaction-based trust and malicious behavior simulation
- **Security-Awareness**: Explicitly models sensitive tasks and malicious node identification
- **Heterogeneity**: Supports four types of devices (High-performance, Standard, Low-power, Storage-optimized) with varied compute/memory/bandwidth profiles
- **Workload Modeling**: Generates Task DAGs with dependency constraints

## Requirements

- Python 3.10+ 
- NumPy

## Usage Example

```python
from edge_sim import EdgeEnvironment, evaluate_assignment

# Initialize the environment 
env = EdgeEnvironment(n_devices=100, n_tasks=80, malicious_ratio=0.15)

# Example: Simple Round-Robin Scheduling
assignment = {task.task_id: task.task_id % env.n_devices for task in env.tasks}

# Evaluate the results [560]
metrics = evaluate_assignment(assignment, env, env.direct_trust)
print(f"Average Trust Score: {metrics['avg_trust']:.4f}")
print(f"Security Score: {metrics['security_score']}%")
