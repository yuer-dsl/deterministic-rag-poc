# Toy Corpus for Deterministic RAG Evaluation
Version: 2025-11  
Author: yuer (Guanyu)

---

## Overview

This file documents the *minimal toy corpus* used for evaluating
deterministic retrieval behavior.  
The goal is **not** to demonstrate performance or scale,  
but to provide a tiny, well-controlled dataset where:

- Global structure is simple  
- Entity relations are explicit  
- Summaries can be checked manually  
- Retrieval is reproducible  
- Any non-deterministic behavior is immediately visible

---

## Corpus Contents

The toy dataset contains **three micro-domains**, each designed to
stress-test a specific retrieval pattern.

### **1. Solar Energy (micro-knowledge domain)**
Documents: `solar_01.txt` – `solar_05.txt`

Topics include:
- photovoltaic basics  
- efficiency factors  
- energy storage concepts  
- common misconceptions  
- future outlook  

Key entities:
`SolarCell`, `Efficiency`, `Storage`, `Inverter`, `GridIntegration`

---

### **2. Urban Transportation (micro-knowledge domain)**
Documents: `transport_01.txt` – `transport_05.txt`

Topics include:
- public transit modes  
- congestion patterns  
- traffic flow theory  
- micro-mobility  
- emissions impact  

Key entities:
`TransitLine`, `FlowRate`, `CarbonImpact`, `EV`, `BikeLane`

---

### **3. Nutrition & Health (micro-knowledge domain)**
Documents: `health_01.txt` – `health_05.txt`

Topics include:
- macro nutrients  
- diet planning  
- calorie density  
- hydration  
- long-term health correlations  

Key entities:
`Protein`, `CalorieDensity`, `Hydration`, `BMI`, `RiskFactor`

---

## Why These Domains?

The three domains are intentionally:

- **non-overlapping**  
- **semantically distinct**  
- **small enough for manual inspection**  
- **rich enough to test clustering and routing**

This makes them ideal for:

### ✔ Testing deterministic global clustering  
### ✔ Validating community summarization  
### ✔ Checking entity-level retrieval correctness  
### ✔ Detecting any sampling-induced drift  
### ✔ Auditing multi-hop or planner-like behavior (should be none)

---

## Deterministic Expectations

Given the corpus size and structure:

- Global clustering should always return **exactly 3 communities**  
- Summaries should be identical across runs (temperature=0)  
- All retrieval operations must be **stable** under repeated queries  
- No multi-hop or planner-style deviation should appear  
- Community summaries should be re-usable as static EMC artifacts  

Any deviation indicates:
- sampling  
- hidden randomness  
- unstable traversal  
- or non-deterministic vector operations

---

## Usage

Recommended pipeline:

1. **Global clustering**  
2. **Summarize each cluster deterministically**  
3. **Freeze summaries as reusable artifacts**  
4. **Perform local semantic retrieval**  
5. **Audit the retrieval result for stability**

This corpus is intentionally small  
so failures can be *understood*, not just *measured*.

---

## Notes

This toy corpus:
- is not intended to evaluate model quality  
- does not represent real-world complexity  
- only exists to stress-test deterministic retrieval paths  

It acts as a microscope—not a benchmark.

---

## License

This toy corpus follows the repository's main LICENSE
(MIT Safe-Use Variant).  
It may be redistributed for educational and experimental purposes.

---

End of file.
