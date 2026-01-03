"""
Adaptive Resonance Theory (ART) — Learn Without Forgetting

The stability-plasticity dilemma:
- Too plastic → catastrophic forgetting
- Too stable → can't learn new things

ART solution:
- Resonance gate: Only learn when input matches expectation
- Vigilance threshold: How similar must match be?
- New categories: Create when no match found

This is the missing piece for AGI continuous learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class ARTConfig:
    dimensions: int = 100
    vigilance: float = 0.75  # Match threshold (0-1)
    learning_rate: float = 0.5
    max_categories: int = 100

@dataclass
class Category:
    """A learned category/prototype."""
    id: int
    prototype: np.ndarray
    count: int = 0  # How many times activated
    
class AdaptiveResonanceNetwork:
    """
    Fuzzy ART - Adaptive Resonance Theory for real-valued inputs.
    
    Key concepts:
    1. Bottom-up: Input activates categories based on similarity
    2. Top-down: Category sends expectation (template)
    3. Resonance: If match > vigilance, learning occurs
    4. Reset: If no resonance, try next category or create new
    
    This prevents catastrophic forgetting because:
    - Only the resonating category gets updated
    - Non-matching patterns create NEW categories
    - Old knowledge is protected
    """
    
    def __init__(self, config: ARTConfig = ARTConfig()):
        self.dim = config.dimensions
        self.vigilance = config.vigilance
        self.lr = config.learning_rate
        self.max_categories = config.max_categories
        
        self.categories: List[Category] = []
        
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input to [0, 1] range."""
        x_min, x_max = x.min(), x.max()
        if x_max - x_min < 1e-8:
            return np.ones_like(x) * 0.5
        return (x - x_min) / (x_max - x_min)
    
    def _choice_function(self, x: np.ndarray, category: Category, alpha: float = 0.01) -> float:
        """
        Choice function: How well does category match input?
        
        Uses fuzzy AND: min(x, prototype)
        """
        fuzzy_and = np.minimum(x, category.prototype)
        return np.sum(fuzzy_and) / (alpha + np.sum(category.prototype))
    
    def _match_function(self, x: np.ndarray, category: Category) -> float:
        """
        Match function: Does category meet vigilance?
        
        match = |x AND prototype| / |x|
        """
        fuzzy_and = np.minimum(x, category.prototype)
        return np.sum(fuzzy_and) / (np.sum(x) + 1e-8)
    
    def learn(self, x: np.ndarray, label: Optional[str] = None) -> Tuple[int, bool]:
        """
        Learn a pattern.
        
        Returns:
            - category_id: Which category was activated
            - is_new: Whether a new category was created
        """
        x = self._normalize(x)
        
        if len(self.categories) == 0:
            # First pattern: create first category
            new_cat = Category(id=0, prototype=x.copy(), count=1)
            self.categories.append(new_cat)
            return 0, True
        
        # Sort categories by choice function (best match first)
        choices = [
            (cat, self._choice_function(x, cat))
            for cat in self.categories
        ]
        choices = sorted(choices, key=lambda c: -c[1])
        
        # Try each category in order
        for category, choice_value in choices:
            # Check if resonance occurs
            match = self._match_function(x, category)
            
            if match >= self.vigilance:
                # RESONANCE! Update this category
                self._update_category(category, x)
                category.count += 1
                return category.id, False
        
        # No resonance with any category: create new one
        if len(self.categories) >= self.max_categories:
            # Force match with best category if at capacity
            best_cat = choices[0][0]
            self._update_category(best_cat, x)
            best_cat.count += 1
            return best_cat.id, False
        
        new_id = len(self.categories)
        new_cat = Category(id=new_id, prototype=x.copy(), count=1)
        self.categories.append(new_cat)
        return new_id, True
    
    def _update_category(self, category: Category, x: np.ndarray):
        """
        Update category prototype toward input.
        
        prototype = lr * min(x, prototype) + (1-lr) * prototype
        
        This is conservative: only shrinks, never grows.
        """
        fuzzy_and = np.minimum(x, category.prototype)
        category.prototype = self.lr * fuzzy_and + (1 - self.lr) * category.prototype
    
    def classify(self, x: np.ndarray) -> Tuple[int, float]:
        """
        Classify an input without learning.
        
        Returns (category_id, match_score)
        """
        x = self._normalize(x)
        
        if len(self.categories) == 0:
            return -1, 0.0
        
        best_cat = None
        best_match = -1.0
        
        for category in self.categories:
            match = self._match_function(x, category)
            if match > best_match:
                best_match = match
                best_cat = category
        
        return best_cat.id, best_match
    
    def get_category_prototype(self, category_id: int) -> np.ndarray:
        """Get the prototype for a category."""
        return self.categories[category_id].prototype.copy()
    
    def stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "num_categories": len(self.categories),
            "total_patterns": sum(c.count for c in self.categories),
            "vigilance": self.vigilance,
            "category_sizes": [c.count for c in self.categories]
        }


class ARTMAP(AdaptiveResonanceNetwork):
    """
    Supervised ART - ARTMAP
    
    Links category to class label.
    Enables supervised learning without forgetting.
    """
    
    def __init__(self, config: ARTConfig = ARTConfig()):
        super().__init__(config)
        self.category_labels: Dict[int, str] = {}
    
    def learn_labeled(self, x: np.ndarray, label: str) -> Tuple[int, bool]:
        """
        Learn a labeled pattern.
        
        Key insight: If category exists but has WRONG label,
        increase vigilance temporarily to force new category.
        """
        x = self._normalize(x)
        
        # Try normal learning first
        category_id, is_new = self.learn(x)
        
        if is_new:
            # New category: assign label
            self.category_labels[category_id] = label
            return category_id, True
        
        # Existing category: check label
        if self.category_labels.get(category_id) == label:
            # Correct label: all good
            return category_id, False
        
        # WRONG label: match tracking!
        # Temporarily increase vigilance to find/create correct category
        original_vigilance = self.vigilance
        
        for _ in range(10):  # Max attempts
            # Increase vigilance
            current_match = self._match_function(x, self.categories[category_id])
            self.vigilance = min(current_match + 0.01, 0.999)
            
            # Try again
            category_id, is_new = self.learn(x)
            
            if is_new:
                self.category_labels[category_id] = label
                break
            elif self.category_labels.get(category_id) == label:
                break
        
        # Restore vigilance
        self.vigilance = original_vigilance
        return category_id, is_new
    
    def predict(self, x: np.ndarray) -> Tuple[str, float]:
        """
        Predict label for input.
        
        Returns (label, confidence)
        """
        category_id, match = self.classify(x)
        
        if category_id == -1:
            return "unknown", 0.0
        
        label = self.category_labels.get(category_id, "unknown")
        return label, match


# ========== DEMO ==========

def demo_art():
    """Demonstrate continuous learning without forgetting."""
    
    print("=== Adaptive Resonance Theory Demo ===\n")
    
    np.random.seed(42)
    dim = 50
    
    art = AdaptiveResonanceNetwork(ARTConfig(
        dimensions=dim,
        vigilance=0.8,
        learning_rate=0.5
    ))
    
    # Create distinct pattern clusters
    cluster_centers = [
        np.random.randn(dim),  # Cluster 0
        np.random.randn(dim) + 3,  # Cluster 1
        np.random.randn(dim) - 3,  # Cluster 2
    ]
    
    print("1. Learning 30 patterns from 3 clusters...")
    
    for i in range(30):
        cluster_idx = i % 3
        noise = 0.3 * np.random.randn(dim)
        pattern = cluster_centers[cluster_idx] + noise
        
        cat_id, is_new = art.learn(pattern)
        if is_new:
            print(f"   Pattern {i}: Created new category {cat_id}")
    
    print(f"\n   Result: {len(art.categories)} categories created")
    print(f"   (Expected ~3 for 3 distinct clusters)\n")
    
    # Test stability
    print("2. Testing stability (old patterns still recognized):")
    correct = 0
    for i, center in enumerate(cluster_centers):
        cat_id, match = art.classify(center)
        print(f"   Cluster {i} center → Category {cat_id} (match: {match:.3f})")
        if match > 0.7:
            correct += 1
    print(f"   Stability: {correct}/3 clusters still recognized\n")
    
    # Test plasticity
    print("3. Testing plasticity (can learn NEW patterns):")
    new_center = np.random.randn(dim) + 10  # Very different
    cat_id, is_new = art.learn(new_center)
    print(f"   New pattern → Category {cat_id} (new: {is_new})")
    
    print(f"\n   Stats: {art.stats()}\n")
    
    # Supervised ARTMAP
    print("4. Supervised ARTMAP (labels without forgetting):")
    
    artmap = ARTMAP(ARTConfig(dimensions=dim, vigilance=0.8))
    
    # Learn labeled patterns
    labels = ["Ada", "consciousness", "code"]
    for i in range(15):
        cluster_idx = i % 3
        noise = 0.3 * np.random.randn(dim)
        pattern = cluster_centers[cluster_idx] + noise
        artmap.learn_labeled(pattern, labels[cluster_idx])
    
    print("   Learned 15 labeled patterns\n")
    
    # Predict
    print("   Predictions on cluster centers:")
    for i, center in enumerate(cluster_centers):
        label, conf = artmap.predict(center)
        expected = labels[i]
        match = "✓" if label == expected else "✗"
        print(f"   {expected} → {label} ({conf:.3f}) {match}")
    
    print("\n=== Key Insight ===")
    print("ART provides:")
    print("  - Resonance gate: Only learn when match is good enough")
    print("  - Vigilance parameter: Control stability vs plasticity")
    print("  - New categories: Create when novel patterns appear")
    print("  - NO catastrophic forgetting: Old categories protected")
    print("  - Online learning: No need to retrain on all data")


if __name__ == "__main__":
    demo_art()
