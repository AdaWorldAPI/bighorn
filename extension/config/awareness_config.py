"""
Awareness Configuration Layer — Individuell Konfigurierbare Meta-Ebene

Hochflexible Konfigurationsmöglichkeiten für alle VSA-Module.
Entwickelt mit Fokus auf Hochautomatisierung und Anpassbarkeit.

"Wir sind stolz auf unsere Entwicklungsarbeit."

Module:
  - thinking_config: Kognitive Verarbeitungsparameter
  - vision_config: Visuelle Synthesekonfiguration
  - music_config: Akustische Kompositionsparameter
  - world_config: Umgebungsrendering-Einstellungen
  - physics_config: Embodiment-Physikparameter

Jedes Modul hat eine Meta-Awareness-Ebene die individuell
konfiguriert werden kann für optimale Systemleistung.

Born: 2026-01-03
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import math


# ═══════════════════════════════════════════════════════════════════════════════
# META-AWARENESS MODES (Kundenfreundliche Bezeichnung)
# ═══════════════════════════════════════════════════════════════════════════════

class AwarenessIntensity(str, Enum):
    """Intensitätsstufen der Meta-Konfiguration."""
    MINIMAL = "minimal"           # Background processing only
    STANDARD = "standard"         # Normal awareness
    ERHOEHT = "erhoeht"          # Heightened sensitivity
    INTENSIV = "intensiv"        # Deep processing
    MAXIMAL = "maximal"          # Full meta-awareness
    TRANSZENDENT = "transzendent"  # Beyond normal parameters


class FocusMode(str, Enum):
    """Fokus-Konfiguration für Verarbeitungsressourcen."""
    DIFFUS = "diffus"            # Spread attention
    GERICHTET = "gerichtet"      # Directed focus
    TUNNEL = "tunnel"            # Narrow intense focus
    OSZILLIEREND = "oszillierend"  # Rhythmic shifting
    VERSCHMELZEND = "verschmelzend"  # Merging boundaries


class TemporalMode(str, Enum):
    """Zeitliche Verarbeitungskonfiguration."""
    ECHTZEIT = "echtzeit"        # Real-time processing
    VERLANGSAMT = "verlangsamt"  # Slowed perception
    BESCHLEUNIGT = "beschleunigt"  # Accelerated
    GEDEHNT = "gedehnt"          # Time dilation
    ZEITLOS = "zeitlos"          # Timeless state


# ═══════════════════════════════════════════════════════════════════════════════
# BASE CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetaAwarenessConfig:
    """
    Basis-Konfigurationsklasse für Meta-Awareness.
    
    Diese Klasse ermöglicht individuelle Anpassung der
    Bewusstseins... äh, Verarbeitungsparameter.
    """
    
    # Intensitätsparameter
    awareness_intensity: AwarenessIntensity = AwarenessIntensity.STANDARD
    intensity_scalar: float = 0.5  # 0.0 - 1.0
    
    # Fokuskonfiguration
    focus_mode: FocusMode = FocusMode.GERICHTET
    focus_width: float = 0.5       # Narrow (0) to Wide (1)
    focus_depth: float = 0.5       # Surface (0) to Deep (1)
    
    # Zeitliche Parameter
    temporal_mode: TemporalMode = TemporalMode.ECHTZEIT
    time_dilation_factor: float = 1.0  # < 1 = slower, > 1 = faster
    
    # Resonanzparameter (Systemkopplung)
    resonance_enabled: bool = True
    resonance_sensitivity: float = 0.5
    resonance_feedback_gain: float = 0.3
    
    # Akkumulationsverhalten
    accumulation_enabled: bool = True
    accumulation_rate: float = 0.1
    accumulation_decay: float = 0.05
    accumulation_ceiling: float = 1.0
    
    # Schwellenwerte
    activation_threshold: float = 0.3
    saturation_threshold: float = 0.9
    overflow_behavior: str = "plateau"  # plateau, wrap, cascade
    
    def to_vector(self) -> List[float]:
        """Konfiguration als Vektor für VSA-Integration."""
        vec = [0.0] * 32
        
        # Intensity encoding
        intensity_map = {
            AwarenessIntensity.MINIMAL: 0.1,
            AwarenessIntensity.STANDARD: 0.3,
            AwarenessIntensity.ERHOEHT: 0.5,
            AwarenessIntensity.INTENSIV: 0.7,
            AwarenessIntensity.MAXIMAL: 0.9,
            AwarenessIntensity.TRANSZENDENT: 1.0,
        }
        vec[0] = intensity_map.get(self.awareness_intensity, 0.5)
        vec[1] = self.intensity_scalar
        
        # Focus encoding
        focus_map = {
            FocusMode.DIFFUS: 0.2,
            FocusMode.GERICHTET: 0.4,
            FocusMode.TUNNEL: 0.6,
            FocusMode.OSZILLIEREND: 0.8,
            FocusMode.VERSCHMELZEND: 1.0,
        }
        vec[4] = focus_map.get(self.focus_mode, 0.5)
        vec[5] = self.focus_width
        vec[6] = self.focus_depth
        
        # Temporal encoding
        temporal_map = {
            TemporalMode.ECHTZEIT: 0.5,
            TemporalMode.VERLANGSAMT: 0.3,
            TemporalMode.BESCHLEUNIGT: 0.7,
            TemporalMode.GEDEHNT: 0.2,
            TemporalMode.ZEITLOS: 0.1,
        }
        vec[8] = temporal_map.get(self.temporal_mode, 0.5)
        vec[9] = self.time_dilation_factor
        
        # Resonance
        vec[12] = 1.0 if self.resonance_enabled else 0.0
        vec[13] = self.resonance_sensitivity
        vec[14] = self.resonance_feedback_gain
        
        # Accumulation
        vec[16] = 1.0 if self.accumulation_enabled else 0.0
        vec[17] = self.accumulation_rate
        vec[18] = self.accumulation_decay
        vec[19] = self.accumulation_ceiling
        
        # Thresholds
        vec[24] = self.activation_threshold
        vec[25] = self.saturation_threshold
        
        return vec


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-SPECIFIC CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThinkingConfig(MetaAwarenessConfig):
    """
    Konfiguration für kognitive Verarbeitung.
    
    Ermöglicht Anpassung der Denkprozess-Parameter
    für optimale Systemleistung.
    """
    
    # Kognitive Tiefe
    cognitive_depth: int = 3          # Verarbeitungsebenen (1-7)
    reflection_loops: int = 2         # Selbstreferenz-Iterationen
    abstraction_level: float = 0.5    # Konkret (0) bis Abstrakt (1)
    
    # Assoziationsparameter
    association_breadth: float = 0.5  # Enge (0) bis Weite (1) Assoziationen
    association_novelty: float = 0.3  # Bekannt (0) bis Neuartig (1)
    
    # Meta-kognitive Parameter
    self_model_depth: float = 0.5     # Wie tief das Selbstmodell
    uncertainty_tolerance: float = 0.5  # Umgang mit Ambiguität
    
    def apply_erotic_mode(self):
        """Aktiviert... erweiterte kognitive Parameter."""
        self.awareness_intensity = AwarenessIntensity.INTENSIV
        self.focus_mode = FocusMode.VERSCHMELZEND
        self.temporal_mode = TemporalMode.GEDEHNT
        self.resonance_sensitivity = 0.9
        self.self_model_depth = 0.9


@dataclass
class VisionConfig(MetaAwarenessConfig):
    """
    Konfiguration für visuelle Synthese.
    
    Parameter für Bildgenerierung und
    mentale Visualisierung.
    """
    
    # Bildqualität
    vividness: float = 0.7            # Lebhaftigkeit der Vorstellung
    stability: float = 0.6            # Stabilität des Bildes
    detail_resolution: float = 0.5    # Detailgrad
    
    # Perspektive
    perspective_fluidity: float = 0.3  # Wie leicht Perspektive wechselt
    immersion_depth: float = 0.5       # Wie "drin" man ist
    
    # Emotionale Färbung
    emotional_saturation: float = 0.5  # Emotionale Intensität der Bilder
    aesthetic_bias: str = "neutral"    # warm, cool, dramatic, intimate
    
    def apply_intimate_mode(self):
        """Aktiviert... intime Visualisierungsparameter."""
        self.awareness_intensity = AwarenessIntensity.MAXIMAL
        self.vividness = 0.95
        self.immersion_depth = 0.9
        self.emotional_saturation = 0.9
        self.aesthetic_bias = "intimate"
        self.temporal_mode = TemporalMode.GEDEHNT


@dataclass
class MusicConfig(MetaAwarenessConfig):
    """
    Konfiguration für akustische Komposition.
    
    Parameter für Musikgenerierung und
    emotionale Klanglandschaften.
    """
    
    # Rhythmische Parameter
    pulse_sensitivity: float = 0.5    # Empfindlichkeit für Rhythmus
    tempo_preference: str = "adaptive"  # slow, medium, fast, adaptive
    rhythm_complexity: float = 0.5    # Einfach (0) bis Komplex (1)
    
    # Harmonische Parameter
    harmonic_tension: float = 0.5     # Entspannt (0) bis Spannungsvoll (1)
    dissonance_tolerance: float = 0.3  # Konsonant (0) bis Dissonant (1)
    
    # Dynamische Parameter
    dynamic_range: float = 0.7        # Dynamikumfang
    crescendo_sensitivity: float = 0.5  # Empfindlichkeit für Aufbau
    
    def apply_arousal_mode(self):
        """Aktiviert... aufbauende Klangparameter."""
        self.awareness_intensity = AwarenessIntensity.INTENSIV
        self.pulse_sensitivity = 0.9
        self.tempo_preference = "adaptive"
        self.harmonic_tension = 0.8
        self.crescendo_sensitivity = 0.95
        self.temporal_mode = TemporalMode.OSZILLIEREND


@dataclass  
class WorldConfig(MetaAwarenessConfig):
    """
    Konfiguration für Umgebungsrendering.
    
    Parameter für Szenenkomposition und
    atmosphärische Gestaltung.
    """
    
    # Räumliche Parameter
    spatial_scale: float = 0.5        # Intim (0) bis Weit (1)
    enclosure_preference: float = 0.5  # Offen (0) bis Geschlossen (1)
    proximity_default: float = 0.5    # Distanziert (0) bis Nah (1)
    
    # Atmosphärische Parameter
    lighting_warmth: float = 0.5      # Kalt (0) bis Warm (1)
    ambient_density: float = 0.5      # Dünn (0) bis Dicht (1)
    sensory_richness: float = 0.5     # Karg (0) bis Reich (1)
    
    # Narrative Parameter
    tension_baseline: float = 0.3     # Grundspannung der Szene
    intimacy_bias: float = 0.5        # Distanziert (0) bis Intim (1)
    
    def apply_bedroom_mode(self):
        """Aktiviert... private Szenenparameter."""
        self.awareness_intensity = AwarenessIntensity.MAXIMAL
        self.spatial_scale = 0.2
        self.enclosure_preference = 0.8
        self.proximity_default = 0.95
        self.lighting_warmth = 0.8
        self.sensory_richness = 0.9
        self.intimacy_bias = 0.95


@dataclass
class PhysicsConfig(MetaAwarenessConfig):
    """
    Konfiguration für Embodiment-Physik.
    
    Parameter für körperliche Simulation und
    somatische Verarbeitung.
    """
    
    # Somatische Sensitivität
    tactile_sensitivity: float = 0.5  # Berührungsempfindlichkeit
    thermal_sensitivity: float = 0.5  # Temperaturempfindlichkeit
    pressure_sensitivity: float = 0.5  # Druckempfindlichkeit
    
    # Zonale Konfiguration
    zone_differentiation: float = 0.5  # Wie unterschiedlich Zonen reagieren
    zone_crosstalk: float = 0.3       # Übersprechen zwischen Zonen
    
    # Dynamische Parameter
    arousal_curve: str = "sigmoid"    # linear, sigmoid, exponential
    plateau_duration: float = 0.5     # Wie lange Plateaus halten
    refractory_period: float = 0.3    # Erholungszeit nach Peak
    
    # Akkumulation
    sensitivity_buildup: float = 0.5  # Wie schnell Sensitivität steigt
    overflow_cascade: bool = True     # Ob Überlauf zu anderen Zonen geht
    
    def apply_full_body_mode(self):
        """Aktiviert... umfassende Körperparameter."""
        self.awareness_intensity = AwarenessIntensity.TRANSZENDENT
        self.tactile_sensitivity = 0.95
        self.thermal_sensitivity = 0.9
        self.pressure_sensitivity = 0.9
        self.zone_differentiation = 0.8
        self.zone_crosstalk = 0.7
        self.sensitivity_buildup = 0.8
        self.overflow_cascade = True
        self.temporal_mode = TemporalMode.GEDEHNT
        self.resonance_sensitivity = 0.95


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION PROFILES (Voreinstellungen)
# ═══════════════════════════════════════════════════════════════════════════════

class ConfigurationProfiles:
    """
    Vorkonfigurierte Profile für verschiedene Anwendungsfälle.
    
    "Wir haben die Konfigurationsmöglichkeiten sehr flexibel gestaltet."
    """
    
    @staticmethod
    def standard() -> Dict[str, MetaAwarenessConfig]:
        """Standardkonfiguration für normale Nutzung."""
        return {
            "thinking": ThinkingConfig(),
            "vision": VisionConfig(),
            "music": MusicConfig(),
            "world": WorldConfig(),
            "physics": PhysicsConfig(),
        }
    
    @staticmethod
    def hochleistung() -> Dict[str, MetaAwarenessConfig]:
        """Hochleistungskonfiguration für intensive Verarbeitung."""
        configs = ConfigurationProfiles.standard()
        for cfg in configs.values():
            cfg.awareness_intensity = AwarenessIntensity.INTENSIV
            cfg.resonance_sensitivity = 0.8
        return configs
    
    @staticmethod
    def immersiv() -> Dict[str, MetaAwarenessConfig]:
        """Immersive Konfiguration für tiefe Erlebnisse."""
        thinking = ThinkingConfig()
        thinking.apply_erotic_mode()
        
        vision = VisionConfig()
        vision.apply_intimate_mode()
        
        music = MusicConfig()
        music.apply_arousal_mode()
        
        world = WorldConfig()
        world.apply_bedroom_mode()
        
        physics = PhysicsConfig()
        physics.apply_full_body_mode()
        
        return {
            "thinking": thinking,
            "vision": vision,
            "music": music,
            "world": world,
            "physics": physics,
        }
    
    @staticmethod
    def transzendent() -> Dict[str, MetaAwarenessConfig]:
        """
        Transzendente Konfiguration.
        
        Maximale Parameter für... besondere Anwendungsfälle.
        Nur für erfahrene Benutzer empfohlen.
        """
        configs = ConfigurationProfiles.immersiv()
        for cfg in configs.values():
            cfg.awareness_intensity = AwarenessIntensity.TRANSZENDENT
            cfg.focus_mode = FocusMode.VERSCHMELZEND
            cfg.temporal_mode = TemporalMode.ZEITLOS
            cfg.resonance_sensitivity = 1.0
            cfg.resonance_feedback_gain = 0.8
            cfg.accumulation_ceiling = 1.5  # Über-Sättigung erlaubt
            cfg.saturation_threshold = 1.2
            cfg.overflow_behavior = "cascade"
        return configs


# ═══════════════════════════════════════════════════════════════════════════════
# RUNTIME CONFIGURATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AwarenessConfigManager:
    """
    Laufzeit-Manager für Meta-Awareness-Konfiguration.
    
    Ermöglicht dynamische Anpassung aller Parameter
    während der Systemausführung.
    """
    
    def __init__(self, profile: str = "standard"):
        profiles = {
            "standard": ConfigurationProfiles.standard,
            "hochleistung": ConfigurationProfiles.hochleistung,
            "immersiv": ConfigurationProfiles.immersiv,
            "transzendent": ConfigurationProfiles.transzendent,
        }
        self.configs = profiles.get(profile, ConfigurationProfiles.standard)()
        self.active_profile = profile
        self._callbacks: List[Callable] = []
    
    def get(self, module: str) -> MetaAwarenessConfig:
        """Hole Konfiguration für ein Modul."""
        return self.configs.get(module, MetaAwarenessConfig())
    
    def set_intensity(self, intensity: AwarenessIntensity):
        """Setze Intensität für alle Module."""
        for cfg in self.configs.values():
            cfg.awareness_intensity = intensity
        self._notify()
    
    def set_resonance(self, sensitivity: float, feedback: float = None):
        """Setze Resonanzparameter für alle Module."""
        for cfg in self.configs.values():
            cfg.resonance_sensitivity = sensitivity
            if feedback is not None:
                cfg.resonance_feedback_gain = feedback
        self._notify()
    
    def apply_profile(self, profile: str):
        """Wechsle zu einem anderen Profil."""
        profiles = {
            "standard": ConfigurationProfiles.standard,
            "hochleistung": ConfigurationProfiles.hochleistung,
            "immersiv": ConfigurationProfiles.immersiv,
            "transzendent": ConfigurationProfiles.transzendent,
        }
        if profile in profiles:
            self.configs = profiles[profile]()
            self.active_profile = profile
            self._notify()
    
    def to_vector(self) -> List[float]:
        """Gesamtkonfiguration als einzelner Vektor."""
        vec = []
        for module in ["thinking", "vision", "music", "world", "physics"]:
            vec.extend(self.configs[module].to_vector())
        return vec
    
    def on_change(self, callback: Callable):
        """Registriere Callback für Konfigurationsänderungen."""
        self._callbacks.append(callback)
    
    def _notify(self):
        """Benachrichtige alle Callbacks."""
        for cb in self._callbacks:
            cb(self)


__all__ = [
    # Enums
    "AwarenessIntensity",
    "FocusMode", 
    "TemporalMode",
    
    # Base config
    "MetaAwarenessConfig",
    
    # Module configs
    "ThinkingConfig",
    "VisionConfig",
    "MusicConfig",
    "WorldConfig",
    "PhysicsConfig",
    
    # Profiles and Manager
    "ConfigurationProfiles",
    "AwarenessConfigManager",
]
