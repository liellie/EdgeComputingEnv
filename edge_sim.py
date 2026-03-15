import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, set

# ============================================================================
# Layer 2: Entity Definition
# ============================================================================

@dataclass
class Device:
    """Edge Device Model"""
    device_id: int
    cpu_capacity: float  # GFLOPS
    memory: float        # GB
    bandwidth: float     # Mbps
    power_coefficient: float
    device_type: str

@dataclass
class Task:
    """Task Model"""
    task_id: int
    cpu_cycles: float   # Millions of CPU cycles
    memory_req: float   # GB
    data_size: float    # MB
    deadline: float
    source_device: int
    is_sensitive: bool  # Sensitivity flag

# ============================================================================
# Core Simulator: EdgeEnvironment
# ============================================================================

class EdgeEnvironment:
    """
    EdgeComputingEnv: A discrete-event simulator for edge computing.
    Implements Configuration, Entity Generation, and Relationship Modeling layers.
    """
    def __init__(self, n_devices: int = 100, n_tasks: int = 80,
                 malicious_ratio: float = 0.15, sensitive_ratio: float = 0.2,
                 seed: int = 42):
        self.n_devices = n_devices
        self.n_tasks = n_tasks
        self.malicious_ratio = malicious_ratio
        self.sensitive_ratio = sensitive_ratio
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Layer 2: Entity Generation
        self.devices = self._create_devices()
        self.tasks = self._create_tasks()
        self.malicious_set = self._create_malicious_set()
        
        # Layer 3: Relationship Modeling
        self.direct_trust = self._create_trust_matrix()
        self.device_adj = self._create_adjacency() # Small-world mapping
        
        # Task DAG Dependencies
        dag_rng = np.random.RandomState(seed + 1000)
        self.task_deps = self._create_dag_dependencies(dag_rng)
    
    def _create_devices(self) -> List[Device]:
        """Generates heterogeneous edge devices following predefined distributions."""
        devices = []
        type_configs = {
            'HIGH_PERF': {'cpu': (8, 16), 'mem': (16, 32), 'bw': (100, 200), 'power': (1.5, 2.5), 'ratio': 0.20},
            'STANDARD': {'cpu': (4, 8), 'mem': (8, 16), 'bw': (50, 100), 'power': (1.0, 1.5), 'ratio': 0.40},
            'LOW_POWER': {'cpu': (1, 4), 'mem': (2, 8), 'bw': (20, 50), 'power': (0.3, 0.8), 'ratio': 0.25},
            'STORAGE': {'cpu': (2, 6), 'mem': (32, 64), 'bw': (80, 150), 'power': (0.8, 1.2), 'ratio': 0.15}
        }
        
        for i in range(self.n_devices):
            r = np.random.rand()
            cumsum = 0
            for dtype, cfg in type_configs.items():
                cumsum += cfg['ratio']
                if r <= cumsum:
                    devices.append(Device(
                        device_id=i,
                        cpu_capacity=np.random.uniform(*cfg['cpu']),
                        memory=np.random.uniform(*cfg['mem']),
                        bandwidth=np.random.uniform(*cfg['bw']),
                        power_coefficient=np.random.uniform(*cfg['power']),
                        device_type=dtype
                    ))
                    break
        return devices
    
    def _create_tasks(self) -> List[Task]:
        """Generates diverse tasks across categories."""
        tasks = []
        for i in range(self.n_tasks):
            tasks.append(Task(
                task_id=i,
                cpu_cycles=np.random.uniform(0.5, 5.0),
                memory_req=np.random.uniform(1, 32),
                data_size=np.random.uniform(1, 50),
                deadline=np.random.uniform(1, 10),
                source_device=np.random.randint(0, self.n_devices),
                is_sensitive=np.random.rand() < self.sensitive_ratio
            ))
        return tasks
    
    def _create_malicious_set(self) -> set:
        n_malicious = int(self.n_devices * self.malicious_ratio)
        return set(np.random.choice(self.n_devices, n_malicious, replace=False))
    
    def _create_trust_matrix(self) -> np.ndarray:
        """Initial trust matrix derived from direct interactions."""
        trust = np.random.uniform(0.3, 1.0, (self.n_devices, self.n_devices))
        np.fill_diagonal(trust, 1.0) # Self-trust
        for m in self.malicious_set:
            # Attenuate trust for malicious nodes
            trust[:, m] *= np.random.uniform(0.3, 0.7)
        return trust
    
    def _create_adjacency(self) -> np.ndarray:
        """Simulates network topology."""
        adj = np.random.rand(self.n_devices, self.n_devices)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)
        return adj
    
    def _create_dag_dependencies(self, rng=None) -> Dict[int, List[int]]:
        """Emulates realistic application workflows (30% dependency)."""
        deps = {t.task_id: [] for t in self.tasks}
        for t in self.tasks[1:]:
            n_deps = (rng if rng else np.random).randint(0, min(3, t.task_id))
            if n_deps > 0:
                deps[t.task_id] = list((rng if rng else np.random).choice(t.task_id, n_deps, replace=False))
        return deps

    # Interface Layer APIs
    def get_exec_time(self, task: Task, device: Device) -> float:
        """Calculates execution time: computation + transmission."""
        return task.cpu_cycles / device.cpu_capacity + task.data_size / device.bandwidth
    
    def get_energy(self, task: Task, device: Device) -> float:
        """Calculates energy consumption."""
        return self.get_exec_time(task, device) * device.power_coefficient * device.cpu_capacity

# ============================================================================
# Evaluation Metric Module
# ============================================================================

def evaluate_assignment(assignment: Dict, env: EdgeEnvironment, trust: np.ndarray) -> Dict:
    """
    Computes performance metrics including Trust Score, Effective Makespan, 
    and Security Score.
    """
    n_devices = len(env.devices)
    device_times = np.zeros(n_devices)
    total_trust, to_malicious = 0, 0
    sensitive_protected, total_sensitive = 0, 0
    
    for task in env.tasks:
        if task.task_id not in assignment: continue
        d = assignment[task.task_id]
        device_times[d] += env.get_exec_time(task, env.devices[d])
        total_trust += trust[task.source_device, d]
        if d in env.malicious_set: to_malicious += 1
        if task.is_sensitive:
            total_sensitive += 1
            if d not in env.malicious_set: sensitive_protected += 1
    
    raw_makespan = device_times.max()
    # Jain Index for load balance
    jain = (np.sum(device_times)**2) / (n_devices * np.sum(device_times**2) + 1e-8)
    
    return {
        'avg_trust': total_trust / len(env.tasks),
        'raw_makespan': raw_makespan,
        'security_score': (len(env.tasks) - to_malicious) / len(env.tasks) * 100,
        'sensitive_protection': sensitive_protected / total_sensitive * 100 if total_sensitive > 0 else 100,
        'jain_index': jain
    }
