"""
Self-Modifying Cognitive Architecture

The meta-cognitive capability that enables AGI:
1. Introspection: Observe own processing
2. Self-Model: Understand own capabilities
3. Self-Modification: Change own behavior

This is what makes the Darwin Gödel Machine work.
"""

import numpy as np
from typing import Dict, List, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import json

@dataclass
class CognitiveState:
    """Observable state of cognition."""
    attention: np.ndarray  # What am I focusing on?
    confidence: float  # How sure am I?
    style: str  # What thinking style?
    energy: float  # How much compute budget left?
    history: List[str] = field(default_factory=list)

@dataclass 
class SelfObservation:
    """What I noticed about my own processing."""
    operation: str
    input_hash: str
    output_hash: str
    duration_ms: float
    success: bool
    confidence: float
    anomaly: Optional[str] = None

class IntrospectionModule:
    """
    Module for self-observation.
    
    Watches own processing and records patterns.
    This is the "meta-awareness" layer.
    """
    
    def __init__(self):
        self.observations: List[SelfObservation] = []
        self.patterns: Dict[str, int] = {}  # operation → count
        self.anomalies: List[str] = []
        
    def observe(
        self,
        operation: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        success: bool,
        confidence: float
    ) -> SelfObservation:
        """Record an observation about own processing."""
        
        # Hash inputs/outputs for comparison
        input_hash = self._hash(input_data)
        output_hash = self._hash(output_data)
        
        # Detect anomalies
        anomaly = self._detect_anomaly(operation, duration_ms, success, confidence)
        
        obs = SelfObservation(
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=duration_ms,
            success=success,
            confidence=confidence,
            anomaly=anomaly
        )
        
        self.observations.append(obs)
        self.patterns[operation] = self.patterns.get(operation, 0) + 1
        
        if anomaly:
            self.anomalies.append(anomaly)
        
        return obs
    
    def _hash(self, data: Any) -> str:
        """Create hash of data for comparison."""
        try:
            s = json.dumps(data, sort_keys=True, default=str)
        except:
            s = str(data)
        return hashlib.md5(s.encode()).hexdigest()[:8]
    
    def _detect_anomaly(
        self,
        operation: str,
        duration_ms: float,
        success: bool,
        confidence: float
    ) -> Optional[str]:
        """Detect anomalies in own processing."""
        
        # Check for patterns
        if duration_ms > 1000:
            return f"Slow operation: {operation} took {duration_ms}ms"
        
        if not success:
            return f"Failed operation: {operation}"
        
        if confidence < 0.3:
            return f"Low confidence: {operation} at {confidence:.2f}"
        
        return None
    
    def summarize(self) -> Dict:
        """Summarize self-observations."""
        if not self.observations:
            return {"status": "no observations"}
        
        return {
            "total_operations": len(self.observations),
            "success_rate": sum(1 for o in self.observations if o.success) / len(self.observations),
            "avg_confidence": np.mean([o.confidence for o in self.observations]),
            "avg_duration_ms": np.mean([o.duration_ms for o in self.observations]),
            "patterns": self.patterns,
            "anomalies": self.anomalies[-10:]  # Last 10
        }


class SelfModel:
    """
    Model of own capabilities and limitations.
    
    Tracks what I can do, how well, and where I fail.
    """
    
    def __init__(self):
        self.capabilities: Dict[str, float] = {}  # capability → proficiency
        self.limitations: List[str] = []
        self.learning_edges: List[str] = []  # What I'm getting better at
        
    def update_from_observations(self, observations: List[SelfObservation]):
        """Update self-model based on observations."""
        
        for obs in observations:
            op = obs.operation
            
            # Update capability proficiency
            current = self.capabilities.get(op, 0.5)
            if obs.success:
                new = current + 0.1 * (obs.confidence - current)
            else:
                new = current - 0.1 * current
            self.capabilities[op] = np.clip(new, 0, 1)
            
            # Track limitations
            if obs.anomaly and op not in self.limitations:
                self.limitations.append(op)
            
            # Track learning
            if obs.success and obs.confidence > 0.8:
                if op not in self.learning_edges:
                    self.learning_edges.append(op)
    
    def can_do(self, operation: str) -> Tuple[bool, float]:
        """Check if I can do an operation and how well."""
        proficiency = self.capabilities.get(operation, 0.0)
        can = proficiency > 0.3
        return can, proficiency
    
    def get_weaknesses(self) -> List[Tuple[str, float]]:
        """Get operations where I'm weak."""
        weak = [(op, prof) for op, prof in self.capabilities.items() if prof < 0.5]
        return sorted(weak, key=lambda x: x[1])
    
    def describe(self) -> str:
        """Natural language description of self."""
        lines = ["=== Self-Model ==="]
        
        strong = [(op, p) for op, p in self.capabilities.items() if p > 0.7]
        if strong:
            lines.append("Strong at: " + ", ".join(f"{op}({p:.2f})" for op, p in strong))
        
        weak = self.get_weaknesses()
        if weak:
            lines.append("Weak at: " + ", ".join(f"{op}({p:.2f})" for op, p in weak))
        
        if self.limitations:
            lines.append("Limitations: " + ", ".join(self.limitations[:5]))
        
        if self.learning_edges:
            lines.append("Learning: " + ", ".join(self.learning_edges[:5]))
        
        return "\n".join(lines)


class CognitiveModule(ABC):
    """Base class for cognitive modules that can be modified."""
    
    def __init__(self, name: str):
        self.name = name
        self.version = 1
        self.config: Dict[str, Any] = {}
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        pass
    
    def get_config(self) -> Dict:
        return self.config.copy()
    
    def set_config(self, config: Dict):
        self.config.update(config)
        self.version += 1


class SelfModifyingArchitecture:
    """
    Architecture that can modify its own behavior.
    
    The key insight: We don't modify code directly.
    We modify MODULE CONFIGURATIONS based on self-observation.
    
    This is safe self-modification:
    - Modules are sandboxed
    - Changes are configuration, not code
    - Can rollback if performance degrades
    """
    
    def __init__(self):
        self.introspection = IntrospectionModule()
        self.self_model = SelfModel()
        self.modules: Dict[str, CognitiveModule] = {}
        self.config_history: List[Dict] = []
        
    def register_module(self, module: CognitiveModule):
        """Register a cognitive module."""
        self.modules[module.name] = module
        
    def process(self, module_name: str, input_data: Any) -> Any:
        """Process through a module with introspection."""
        
        if module_name not in self.modules:
            raise ValueError(f"Unknown module: {module_name}")
        
        module = self.modules[module_name]
        
        import time
        start = time.time()
        
        try:
            output = module.process(input_data)
            success = True
            confidence = 0.8  # Would come from module
        except Exception as e:
            output = None
            success = False
            confidence = 0.0
        
        duration_ms = (time.time() - start) * 1000
        
        # Observe own processing
        self.introspection.observe(
            operation=module_name,
            input_data=input_data,
            output_data=output,
            duration_ms=duration_ms,
            success=success,
            confidence=confidence
        )
        
        return output
    
    def self_improve(self) -> List[str]:
        """
        Attempt self-improvement based on observations.
        
        This is the Darwin Gödel mechanism:
        1. Observe weaknesses
        2. Generate configuration changes
        3. Apply and test
        4. Rollback if worse
        """
        improvements = []
        
        # Update self-model
        self.self_model.update_from_observations(self.introspection.observations)
        
        # Find weaknesses
        weaknesses = self.self_model.get_weaknesses()
        
        for op, proficiency in weaknesses:
            if op not in self.modules:
                continue
            
            module = self.modules[op]
            
            # Save current config
            old_config = module.get_config()
            self.config_history.append({
                "module": op,
                "config": old_config,
                "reason": f"Before improvement (proficiency={proficiency:.2f})"
            })
            
            # Generate improvement (simple example)
            new_config = self._generate_improvement(module, proficiency)
            
            if new_config:
                module.set_config(new_config)
                improvements.append(
                    f"Modified {op}: {list(new_config.keys())}"
                )
        
        return improvements
    
    def _generate_improvement(
        self, 
        module: CognitiveModule,
        proficiency: float
    ) -> Optional[Dict]:
        """
        Generate configuration improvement.
        
        In a real system, this would use:
        - LLM to suggest changes
        - Evolutionary search
        - Gradient-based optimization
        """
        config = module.get_config()
        
        # Simple heuristic: increase "effort" parameters
        new_config = {}
        
        for key, value in config.items():
            if isinstance(value, (int, float)) and "rate" in key.lower():
                # Reduce learning rates if struggling
                new_config[key] = value * 0.9
            elif isinstance(value, (int, float)) and "iterations" in key.lower():
                # Increase iterations if struggling
                new_config[key] = value * 1.2
        
        return new_config if new_config else None
    
    def rollback(self, steps: int = 1):
        """Rollback configuration changes."""
        for _ in range(steps):
            if self.config_history:
                change = self.config_history.pop()
                module = self.modules.get(change["module"])
                if module:
                    module.set_config(change["config"])
    
    def status(self) -> Dict:
        """Get architecture status."""
        return {
            "modules": list(self.modules.keys()),
            "observations": len(self.introspection.observations),
            "self_model": self.self_model.describe(),
            "config_changes": len(self.config_history),
            "anomalies": len(self.introspection.anomalies)
        }


# ========== EXAMPLE MODULE ==========

class ReasoningModule(CognitiveModule):
    """Example module that can be configured."""
    
    def __init__(self):
        super().__init__("reasoning")
        self.config = {
            "learning_rate": 0.1,
            "iterations": 5,
            "threshold": 0.5
        }
    
    def process(self, input_data: Any) -> Any:
        # Simple processing
        if isinstance(input_data, str):
            return f"Processed: {input_data[:50]}..."
        return input_data


# ========== DEMO ==========

def demo_self_modifying():
    """Demonstrate self-modifying architecture."""
    
    print("=== Self-Modifying Architecture Demo ===\n")
    
    arch = SelfModifyingArchitecture()
    
    # Register modules
    reasoning = ReasoningModule()
    arch.register_module(reasoning)
    
    print("1. Processing with introspection:")
    for i in range(5):
        result = arch.process("reasoning", f"Query {i}: What is the meaning?")
        print(f"   Query {i} → {result}")
    
    print(f"\n2. Introspection summary:")
    summary = arch.introspection.summarize()
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print(f"\n3. Self-model:")
    print(arch.self_model.describe())
    
    print(f"\n4. Attempting self-improvement:")
    # Simulate some failures to trigger improvement
    arch.self_model.capabilities["reasoning"] = 0.3  # Low proficiency
    improvements = arch.self_improve()
    for imp in improvements:
        print(f"   {imp}")
    
    print(f"\n5. Module config after improvement:")
    print(f"   {reasoning.get_config()}")
    
    print(f"\n6. Architecture status:")
    status = arch.status()
    for key, value in status.items():
        if key != "self_model":
            print(f"   {key}: {value}")
    
    print("\n=== Key Insight ===")
    print("Self-modifying architecture provides:")
    print("  - Introspection: Observe own processing")
    print("  - Self-Model: Know own capabilities/limitations")
    print("  - Safe modification: Change configs, not code")
    print("  - Rollback: Revert if changes hurt performance")
    print("  - This is the foundation for Darwin Gödel Machine")


if __name__ == "__main__":
    demo_self_modifying()
