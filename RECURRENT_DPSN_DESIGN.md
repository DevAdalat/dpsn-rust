# Enhancing DPSN: The "Recurrent-DPSN" Architecture (R-DPSN)

## Executive Summary
To enable the **Dynamic Parameter Selection Network (DPSN)** to compete with dense 10B+ parameter models in complex reasoning tasks, we propose an architectural evolution: **Recurrent-DPSN**.

By implementing a **Recurrent Execution Loop** (Universal Transformer pattern), we decouple the model's **Depth (Reasoning)** from its **Size (Storage)**. This allows a DPSN to utilize its massive parameter pool sequentially, mimicking the deep layer stacks of dense LLMs without the memory cost of duplicating weights.

---

## 1. The Core Limitation of Standard DPSN
**Current State:** The standard DPSN operates as a "Wide but Shallow" network.
*   **Width:** Massive knowledge capacity (10B+ pool).
*   **Depth:** Single-step retrieval. The router looks at the input once, grabs parameters, and computes.

**The Problem:** Deep reasoning requires **sequential processing**. Dense models achieve this by physically stacking 40-80 layers. A single pass—even with the perfect parameters—cannot solve a problem that requires multiple logical steps (e.g., `A -> B -> C`).

## 2. The Solution: Recurrent Weight Reuse
Instead of adding physical layers (which increases VRAM usage), we iterate the input state through the **same** Router and Pool multiple times.

### 2.1 The Concept
1.  **Shared Workspace:** The Parameter Pool is no longer just a "layer"; it is a shared library of functions.
2.  **Iterative Refinement:**
    *   **Loop 1 (Surface):** The Router identifies basic syntax. It fetches "Grammar Parameters" from the Pool. State updates.
    *   **Loop 2 (Context):** The Router sees the updated state. It realizes the context is "Code". It fetches "Python Syntax Parameters". State updates.
    *   **Loop 3 (Logic):** The Router sees code context. It fetches "Algorithm Parameters".
3.  **Result:** The model simulates an infinite-depth network by dynamically recomposing itself at every step.

---

## 3. Detailed Architecture Specifications

### 3.1 Configuration Parameters
To match a 10B Dense Model (e.g., Llama-3-8B) in reasoning capability:

| Parameter | Value | Justification |
| :--- | :--- | :--- |
| **Pool Size ($M$)** | **10 Billion** | Matches the "Knowledge Capacity" of the dense rival. |
| **Active Budget ($k$)** | **250M - 500M** | Keep active compute low (speed). |
| **Recurrence Steps ($T$)** | **16 - 32** | Matches the "Depth/Reasoning" of the dense rival (Llama-3 has 32 layers). |
| **Effective Compute** | **$k \times T$** | $250M \times 32 = 8 \text{ Billion}$. The model "thinks" as much as a dense model. |

### 3.2 The Algorithm (Pseudocode)

```rust
fn forward(input: Tensor) -> Tensor {
    let mut x = input; // State vector
    
    // The "Thinking" Loop
    for t in 0..RECURRENCE_STEPS {
        
        // 1. Positional Awareness (Optional but recommended)
        // Add a "Step Embedding" so the model knows "I am in step 5 of 32"
        let step_embedding = get_step_embedding(t); 
        let current_state = x + step_embedding;

        // 2. Attention Block (Standard or Shared)
        // Can share one Attention block across all steps, or have N unique ones.
        // Sharing is recommended to keep parameter count pure.
        let attn_out = self.attention.forward(current_state.clone());
        x = x + attn_out; // Residual 1

        // 3. Dynamic Router (The Brain)
        // The router sees the EVOLVED state 'x'
        // It decides: "Given what I computed in step t-1, what do I need now?"
        let (indices, weights) = self.router.forward(x.clone());
        
        // 4. Sparse Execution (The Body)
        // Fetches completely different parameters than step t-1
        let ffn_out = self.execution_engine.forward(x.clone(), weights);
        
        // 5. Update State
        x = x + ffn_out; // Residual 2
    }
    
    return self.final_norm(x);
}
```

## 4. Key Advantages of Recurrent-DPSN

### 4.1 "Turing Complete" Routing
In a standard stack, Layer 10 *must* follow Layer 9.
In Recurrent-DPSN, the Router can choose to access "Basic Grammar" weights at Step 1, "Complex Logic" at Step 2, and then return to "Basic Grammar" at Step 3 if needed. It is a **non-linear control flow**.

### 4.2 Adaptive Compute Depth (Adaptive Thinking)
We can implement an **Early Exit** mechanism. The router can output a `halt_probability`.
*   **Simple Token ("the"):** Router signals "Halt" after 2 loops. (Fast inference)
*   **Hard Token ("Quantum"):** Router loops for the full 32 steps. (Deep thought)
This allows the model to have a variable depth per token, maximizing speed without sacrificing peak IQ.

### 4.3 Parameter Efficiency
A Dense 10B model stores 32 separate FFN layers.
*   Layer 1 weights are used ONLY for low-level features.
*   Layer 32 weights are used ONLY for high-level outputs.
*   This is wasteful.

The Recurrent-DPSN stores 10B parameters in ONE pool.
*   Any parameter can be used at ANY step.
*   High-level logic weights can be accessed early if the prompt demands it.
*   Low-level syntax weights can be reused late for formatting.

## 5. Implementation Roadmap

1.  **Refactor `DPSN` Struct:**
    *   Add `recurrence_steps: usize` to `DPSNConfig`.
    *   Change `forward` method to include a loop.
    
2.  **Add Step Embeddings:**
    *   Implement a small embedding table `[max_steps, embed_dim]`.
    *   Inject this into the input at the start of each loop iteration. This is crucial so the router doesn't get stuck in a loop doing the same thing.

3.  **Adaptive Halting (Advanced):**
    *   Add a small head to the Router: `Linear(hidden) -> Sigmoid(halt_prob)`.
    *   If `halt_prob > threshold`, break the loop.

---
**Verdict:** This change transforms DPSN from a "Efficient Memory Lookup" system into a true "Reasoning Engine" capable of competing with state-of-the-art dense models.
