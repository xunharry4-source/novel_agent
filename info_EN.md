# Omni-Galaxy Protocol (PGA) Worldview Management Protocol

This document defines the core structural rules (0-4 Architecture) for PGA worldview management. All Agent generation, auditing, and modification logic must strictly adhere to this protocol.

## 0. Definitions

*   **PGA Protocols**: Omni-Galaxy Protocols, the fundamental physical rules governing all civilizations and technological paths.
*   **Ecological Niches**: The survival space for life and factions in each sector, determined by energy output (e.g., star type, Dyson structures).
*   **Thermodynamics Lore**: The bottom-layer logic, based on the laws of entropy and energy conservation.
*   **T**: The iteration cycle or the moment a setting is established.

---

## 1. Entry Logic

**Core Principle: Any new element (race, faction, technology) must pass "Sector Ecological Adaptation" verification before entering the database.**

*   **Prior Verification**: New settings must prove physical feasibility in a specific sector.
    *   Is the energy source clearly defined?
    *   Does it comply with the entropy balance of the sector?
*   **Category Alignment**: New entries must belong to one of {Races, Geographies, Factions, Mechanisms, History} and match the existing definitions in that category.
*   **Rejection Mechanism**: Settings that cannot prove physical rationality or cause logic spillover are prohibited.

---

## 2. Conflict & Modification Logic

**Core Principle: Any modification to existing settings must resolve logical contradictions without violating bottom-layer thermodynamic rules.**

*   **Change Constraints**: When modifying a "Race", related "Factions" or "History" must synchronously calculate loss/gain to ensure total energy conservation.
*   **Conflict Decision**: If a new setting conflicts with the primary Thermodynamics lore, the new setting is invalid.
*   **Logic Closure**: Modifications must pass the `reviewer` node's logical audit before being established.

---

## 3. Priority & Exclusivity

**The system follows this priority order (Highest to Lowest):**

1.  **Thermodynamic Iron Laws**: Entropy and energy conservation (absolute veto power).
2.  **PGA Core Prohibitions**: Prohibit time control, high-dimensional gods, future prediction, reality modification, and infinite energy.
3.  **PGA Base Protocols**: Cross-sector common technical and social standards.
4.  **Regional Details**: Localized backgrounds for specific planets or factions.

*Exclusivity*: High-priority rules directly override or block modification attempts from lower-priority rules.

---

## 4. Independence & Modularity

*   **Data Isolation**: Updates to multiple datasets (Race, Faction, Mechanism) must remain modular. Modifying race A's physiology should not directly change sector B's geographic constants.
*   **Routing Consistency**: The audit process must use the initially defined category logic and cannot switch rule templates midway.
*   **No Spillover**: Impact of adjustments to a single entity must be limited to known correlation chains.

---

## 5. Observability & Quality Assurance

**Core Principle: All agent executions must be measurable, traceable, and auditable.**

*   **Full-Stack Tracing**: All LLM calls must be traced via LangFuse to ensure any logical drift can be backtracked to its source.
*   **Real-time Monitoring**: Production environments must monitor token usage and request latency via Prometheus.
*   **Exception Alerting**: Any unhandled runtime exception must be automatically reported to Sentry.
*   **Token Governance**: Monitor `llm_token_usage_total` to optimize prompt structures for economic long-form creation.

---

## 6. Highest Prohibitions

> [!CAUTION]
> **Absolute Prohibitions**:
> 1. No time travel or time control of any form.
> 2. No "omniscient/omnipotent" or high-dimensional gods (all beings limited by physical scale).
> 3. No absolute prediction of the future.
> 4. No reality micro-adjustments crossing thermodynamic thresholds.
> 5. No infinite energy devices.
