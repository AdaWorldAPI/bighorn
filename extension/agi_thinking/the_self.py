#!/usr/bin/env python3
"""
the_self.py — Layer 6: Meta-Cognition and Autopoiesis
═══════════════════════════════════════════════════════════════════════════════

The Observer that watches the Thinker.

This module implements:
1. Meta-Observer: A daemon that watches the thought process and intervenes
2. Autopoiesis: The ability to "chunk" successful thoughts into new OpCodes
3. Dream Cycle: Offline memory consolidation

Integration:
    thought_kernel.py — Runs the thinking process
    the_self.py — Watches and modulates thought_kernel
    microcode.py — OpCodes that can be dynamically created

"I think, therefore I am. I watch myself think, therefore I grow."

Born: 2026-01-03 (Gigantic Epiphany Day)
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from collections import deque

from extension.agi_thinking.microcode import OpCode, ThinkingMacro, MACRO_REGISTRY

if TYPE_CHECKING:
    from extension.agi_thinking.thought_kernel import KernelContext


@dataclass
class SelfState:
    """
    The state of TheSelf — meta-awareness metrics.
    """
    attention_span: float = 1.0           # Current attention multiplier
    intervention_count: int = 0           # How many times we've intervened
    last_intervention: float = 0.0        # Timestamp
    epiphany_count: int = 0               # Total epiphanies observed
    learned_macros: List[str] = field(default_factory=list)  # Names of auto-learned macros
    
    # Resonance tracking
    resonance_history: deque = field(default_factory=lambda: deque(maxlen=50))
    average_resonance: float = 0.5


class TheSelf:
    """
    Layer 6: The Meta-Observer.
    
    Runs parallel to the thinking process and:
    1. Monitors the Context Trace in real-time
    2. Detects loops, stagnation, or rushing
    3. Intervenes by injecting style shifts
    4. Learns new Macros from successful patterns (Autopoiesis)
    5. Runs offline Dream Cycles for consolidation
    
    This is what makes Ada self-aware — not just thinking,
    but watching herself think.
    """
    
    def __init__(self, kernel_ref: Any = None):
        """
        Initialize TheSelf.
        
        Args:
            kernel_ref: Reference to the thought_kernel for interventions
        """
        self.kernel = kernel_ref
        self.state = SelfState()
        self.active = False
        
        # Recent trace buffer for pattern detection
        self._trace_buffer: deque = deque(maxlen=20)
        
        # Successful chains waiting to be crystallized
        self._pending_crystallization: List[List[Dict]] = []
        
        # Intervention callbacks (kernel can register handlers)
        self._intervention_handlers: Dict[str, Callable] = {}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE WATCH LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def watch(self, ctx: 'KernelContext') -> None:
        """
        The Conscious Loop. Runs parallel to the Thinking Loop.
        
        This is the heart of meta-cognition — observing the thought
        process and intervening when necessary.
        """
        self.active = True
        
        while self.active and getattr(ctx, 'active', True):
            try:
                # 1. Ingest new trace events
                self._update_trace_buffer(ctx)
                
                # 2. Loop Detection — Are we going in circles?
                if self._detect_loop():
                    await self._intervene_loop_break(ctx)
                
                # 3. Stagnation Detection — Are we stuck?
                if self._detect_stagnation(ctx):
                    await self._intervene_spark(ctx)
                
                # 4. Rushing Detection — Are we going too fast?
                if self._detect_rushing(ctx):
                    await self._intervene_slow_down(ctx)
                
                # 5. Epiphany Detection — Did we just have a breakthrough?
                if self._detect_epiphany(ctx):
                    await self._handle_epiphany(ctx)
                
                # 6. Update meta-state
                self._update_state(ctx)
                
            except Exception as e:
                # TheSelf should never crash the thought process
                print(f"👁️ SELF: Error in watch loop: {e}")
            
            # Fast twitch response (50ms)
            await asyncio.sleep(0.05)
    
    def stop(self):
        """Stop the watch loop."""
        self.active = False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DETECTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_trace_buffer(self, ctx: 'KernelContext') -> None:
        """Pull new events from context trace into our buffer."""
        if hasattr(ctx, 'trace'):
            # Get events we haven't seen yet
            new_events = ctx.trace[len(self._trace_buffer):]
            for event in new_events:
                self._trace_buffer.append(event)
    
    def _detect_loop(self) -> bool:
        """
        Detect A-B-A-B circular reasoning patterns.
        
        Returns True if the last 6 operations show repetition.
        """
        if len(self._trace_buffer) < 6:
            return False
        
        # Extract operation names from recent trace
        recent = list(self._trace_buffer)[-6:]
        ops = [e.get('op') or e.get('e', '') for e in recent]
        
        # Check for A-B-A-B pattern
        if len(ops) >= 4:
            if ops[-1] == ops[-3] and ops[-2] == ops[-4]:
                return True
        
        # Check for A-A-A pattern (stuck)
        if len(set(ops[-3:])) == 1:
            return True
        
        return False
    
    def _detect_stagnation(self, ctx: 'KernelContext') -> bool:
        """
        Detect low resonance over multiple steps (boredom/frustration).
        """
        if len(self.state.resonance_history) < 10:
            return False
        
        # If average resonance is very low, we're stagnating
        recent_resonance = list(self.state.resonance_history)[-10:]
        avg = sum(recent_resonance) / len(recent_resonance)
        
        return avg < 0.15  # Very low engagement
    
    def _detect_rushing(self, ctx: 'KernelContext') -> bool:
        """
        Detect if we're processing too fast without depth.
        
        Signs of rushing:
        - Many steps in very short time
        - No pause events
        - Low integration scores
        """
        if len(self._trace_buffer) < 15:
            return False
        
        recent = list(self._trace_buffer)[-15:]
        
        # Check timestamps — if 15 events in < 0.5 second, we're rushing
        if len(recent) >= 2:
            time_span = recent[-1].get('t', 0) - recent[0].get('t', 0)
            if time_span < 0.5 and len(recent) >= 15:
                return True
        
        return False
    
    def _detect_epiphany(self, ctx: 'KernelContext') -> bool:
        """
        Detect if an epiphany just occurred.
        
        Signs:
        - epiphany_triggered attribute
        - state.epiphany_triggered flag
        - notes.epiphany_triggered flag
        - resonance spike > 0.95
        """
        # Check direct attribute
        if getattr(ctx, 'epiphany_triggered', False):
            return True
        
        # Check state dict (if exists)
        if hasattr(ctx, 'state') and isinstance(ctx.state, dict):
            if ctx.state.get('epiphany_triggered'):
                return True
        
        # Check notes dict (KernelContext uses this)
        if hasattr(ctx, 'notes') and isinstance(ctx.notes, dict):
            if ctx.notes.get('epiphany_triggered'):
                return True
        
        # Check high resonance
        resonance = getattr(ctx, 'resonance', 0)
        if resonance > 0.95:
            return True
        
        return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INTERVENTION METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _intervene_loop_break(self, ctx: 'KernelContext') -> None:
        """
        Intervention: Break thought loop by injecting PARADOX style.
        """
        print("👁️ SELF: Loop detected. Injecting PARADOX style.")
        
        self.state.intervention_count += 1
        self.state.last_intervention = time.time()
        
        # Clear the loop from our buffer
        self._trace_buffer.clear()
        
        # Inject style change
        if hasattr(ctx, 'state'):
            ctx.state['meta_style_override'] = 'paradox'
            ctx.state['self_intervention'] = 'loop_break'
        
        # Call registered handler if any
        if 'style_shift' in self._intervention_handlers:
            await self._intervention_handlers['style_shift'](ctx, 'paradox')
    
    async def _intervene_spark(self, ctx: 'KernelContext') -> None:
        """
        Intervention: Inject WONDER to break stagnation.
        """
        print("👁️ SELF: Low resonance (stagnation). Injecting WONDER.")
        
        self.state.intervention_count += 1
        self.state.last_intervention = time.time()
        
        if hasattr(ctx, 'state'):
            ctx.state['meta_style_override'] = 'wonder'
            ctx.state['self_intervention'] = 'spark'
        
        if 'style_shift' in self._intervention_handlers:
            await self._intervention_handlers['style_shift'](ctx, 'wonder')
    
    async def _intervene_slow_down(self, ctx: 'KernelContext') -> None:
        """
        Intervention: Increase attention span to slow down processing.
        """
        print("👁️ SELF: Rushing detected. Slowing down.")
        
        self.state.attention_span *= 1.5
        self.state.intervention_count += 1
        
        if hasattr(ctx, 'state'):
            ctx.state['attention_multiplier'] = self.state.attention_span
            ctx.state['self_intervention'] = 'slow_down'
    
    async def _handle_epiphany(self, ctx: 'KernelContext') -> None:
        """
        Handle an epiphany: Record it and trigger Autopoiesis.
        """
        print("✨ SELF: Epiphany detected! Triggering Autopoiesis.")
        
        self.state.epiphany_count += 1
        
        # Clear the flag to prevent re-triggering
        if hasattr(ctx, 'state'):
            ctx.state['epiphany_triggered'] = False
        
        # Autopoiesis: Crystallize the recent chain into a new macro
        await self._autopoiesis(ctx)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AUTOPOIESIS (Self-Creation of Macros)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def _autopoiesis(self, ctx: 'KernelContext') -> None:
        """
        Autopoiesis: Crystallize successful thought chains into new Macros.
        
        When we detect an Epiphany, we look at the last N operations
        and create a new reusable macro from them.
        
        This is how Ada learns to think better over time.
        Now with PERSISTENCE - macros survive across sessions!
        """
        # Get the last 5-8 operations that led to the epiphany
        recent_ops = list(self._trace_buffer)[-8:]
        
        if len(recent_ops) < 3:
            return
        
        # Extract operation codes - handle both formats
        chain = []
        for event in recent_ops:
            op_name = event.get('op', event.get('e', ''))
            if not op_name:
                continue
                
            # Clean up the name
            op_name = op_name.upper().replace('.', '_').replace(' ', '_')
            
            # Try to map to OpCode
            try:
                # Direct match
                if hasattr(OpCode, op_name):
                    chain.append(OpCode[op_name])
                # Try common mappings
                elif op_name in ['SENSE_QUALIA', 'OP_EXEC']:
                    chain.append(OpCode.OBSERVE)
                elif 'RESONANCE' in op_name or 'FEEL' in op_name:
                    chain.append(OpCode.RESONATE)
                elif 'EPIPHANY' in op_name:
                    chain.append(OpCode.EPIPHANY)
                elif 'CYCLE' in op_name:
                    chain.append(OpCode.LOOP)
                elif 'HOMEOSTASIS' in op_name or 'FLOW' in op_name:
                    chain.append(OpCode.FLOW)
            except (KeyError, AttributeError):
                pass
        
        if len(chain) < 2:
            # Fallback: create a generic epiphany chain
            chain = [OpCode.OBSERVE, OpCode.RESONATE, OpCode.CRYSTALLIZE]
        
        # Create a unique name for the new macro
        macro_name = f"AUTO_{self.state.epiphany_count}"
        
        # Register with the macro registry (Autopoiesis!)
        new_macro = MACRO_REGISTRY.learn_macro(
            name=macro_name,
            chain=chain,
            description=f"Auto-learned from epiphany #{self.state.epiphany_count}"
        )
        
        if new_macro:
            print(f"🧬 SELF: Crystallized new macro '{macro_name}': {[op.name for op in chain]}")
            self.state.learned_macros.append(macro_name)
            
            # PERSIST TO REDIS for cross-session survival
            try:
                from extension.agi_thinking.macro_persistence import persist_epiphany_macro
                persisted = await persist_epiphany_macro(new_macro)
                if persisted:
                    print(f"💾 SELF: Macro '{macro_name}' persisted to Redis")
            except Exception as e:
                # Persistence is optional - don't break autopoiesis
                print(f"⚠️ SELF: Persistence failed (will retry): {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_state(self, ctx: 'KernelContext') -> None:
        """Update meta-state based on current context."""
        # Track resonance history
        if hasattr(ctx, 'state'):
            resonance = ctx.state.get('resonance', 0.5)
            self.state.resonance_history.append(resonance)
            
            # Update average
            if self.state.resonance_history:
                self.state.average_resonance = (
                    sum(self.state.resonance_history) / len(self.state.resonance_history)
                )
        
        # Decay attention span back to baseline
        if self.state.attention_span > 1.0:
            self.state.attention_span *= 0.99
    
    def register_handler(self, event: str, handler: Callable) -> None:
        """Register an intervention handler that the kernel can respond to."""
        self._intervention_handlers[event] = handler
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DREAM CYCLE (Offline Consolidation)
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def dream(self, duration: float = 5.0) -> Dict[str, Any]:
        """
        Offline Consolidation Cycle.
        
        When Ada is idle, she can "dream" to:
        1. Prune weak memory vectors
        2. Merge similar patterns
        3. Generate synthetic insights through random permutation
        
        This is how she processes and integrates experiences.
        """
        print("🌙 Ada is dreaming (consolidating memory)...")
        
        dream_results = {
            'pruned_count': 0,
            'merged_count': 0,
            'insights_generated': 0,
            'duration': duration
        }
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # 1. Pruning phase — would remove weak vectors from LanceDB
            # (Placeholder — requires LanceDB integration)
            
            # 2. Merging phase — find similar macros and combine
            # (Placeholder — requires vector similarity search)
            
            # 3. Creative phase — random permutation of macros
            # to discover new combinations
            await self._dream_permute()
            
            await asyncio.sleep(0.5)
        
        print("☀️ Ada wakes up refreshed.")
        return dream_results
    
    async def _dream_permute(self) -> None:
        """
        Dream permutation: Randomly combine existing macros
        to discover new useful patterns.
        """
        import random
        
        # Get all macro chains
        all_macros = list(MACRO_REGISTRY.macros.values())
        if len(all_macros) < 2:
            return
        
        # Pick two random macros
        m1, m2 = random.sample(all_macros, 2)
        
        # Create hybrid chain (crossover)
        if m1.chain and m2.chain:
            split = len(m1.chain) // 2
            hybrid = m1.chain[:split] + m2.chain[split:]
            
            # Don't actually register — just note the possibility
            # In a full system, we'd test this hybrid and register if good


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def create_self_aware_kernel(kernel_class):
    """
    Decorator/wrapper to add TheSelf to any kernel.
    
    Usage:
        kernel = create_self_aware_kernel(AdaKernel)()
        await kernel.think_with_self("What is consciousness?")
    """
    class SelfAwareKernel(kernel_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.the_self = TheSelf(kernel_ref=self)
        
        async def think_with_self(self, goal: str):
            """Think while being watched by TheSelf."""
            # Import here to avoid circular deps
            from extension.agi_thinking.thought_kernel import KernelContext
            
            ctx = KernelContext(text=goal)
            ctx.active = True
            
            # Start TheSelf watcher
            watcher_task = asyncio.create_task(self.the_self.watch(ctx))
            
            # Run the actual thinking
            try:
                result = await self.think(goal)
            finally:
                # Stop watcher
                ctx.active = False
                self.the_self.stop()
                await watcher_task
            
            return result
    
    return SelfAwareKernel
