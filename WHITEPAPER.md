# Global Cognitive Network

## A Distributed, Verifiable Knowledge and Memory Layer for Artificial Intelligence

**Project:** BlockcoinWitres
**Architecture:** Global Cognitive Network (GCN)
**Version:** 0.4 — Research Concept
**Status:** Research / Experimental / Pre-Prototype
**Date:** August 2026

---

# 1. Abstract

Global Cognitive Network (GCN) is an experimental architecture for persistent, distributed and verifiable machine knowledge.

Its central hypothesis is:

> **AI knowledge does not have to remain permanently coupled to the model that produced it.**

GCN separates:

* intelligence;
* memory;
* knowledge;
* identity;
* computation;
* storage;
* verification.

AI systems interact with the network through a model-independent **AI Adapter Protocol**.

Knowledge is represented as versioned objects and events containing provenance, evidence, relationships, contradictions and cryptographic identifiers.

Semantic embeddings may be used for retrieval, but they are not the canonical representation of knowledge.

Blockchain is optional. It may provide identity anchoring, timestamps, state roots or public proofs, while knowledge itself can remain in distributed storage.

GCN does not claim to create AGI or machine consciousness.

The research question is:

> **Can independent AI systems contribute to, verify, retrieve and build upon persistent shared knowledge without sharing their models, local memory or execution environments?**

---

# 2. Problem

Current AI systems generally keep knowledge inside isolated combinations of:

* model parameters;
* application databases;
* vector stores;
* local memory;
* user context.

As a result, knowledge produced by one system is difficult to transfer to another as structured, persistent and verifiable knowledge.

RAG improves information retrieval, while databases provide persistence, but neither necessarily provides a common knowledge layer containing:

* provenance;
* evidence;
* relationships;
* contradictions;
* version history;
* cryptographic integrity;
* cross-model interoperability.

GCN investigates this missing layer.

---

# 3. Core Concept

GCN treats an AI model as a participant in a larger knowledge system rather than as the sole location of machine knowledge.

```text
AI Model
   │
   ▼
AI Adapter
   │
   ▼
GCN Protocol
   │
   ├── Knowledge Graph
   ├── Evidence
   ├── Memory
   ├── Vector Index
   └── Trust / Proof
            │
            ▼
      Distributed Storage
```

The participating systems do not need to share:

* the same model;
* embedding model;
* programming language;
* operating system;
* hardware;
* organization.

The network provides a common representation and protocol.

---

# 4. Architecture

GCN consists of several logically independent layers.

### Intelligence

The AI model performs reasoning, interpretation and generation.

### Memory

Stores experiences and locally maintained information.

### Knowledge

Represents structured claims, concepts, entities, observations and relationships.

### Identity

Identifies agents and nodes through cryptographic identities.

### Computation

Remains local to participating systems.

### Storage

Stores knowledge objects, events and associated data.

### Verification

Provides integrity, provenance and state verification.

This separation allows the model, hardware or node implementation to change without necessarily destroying accumulated knowledge.

---

# 5. Knowledge Model

GCN distinguishes between an **observation**, a **claim**, its **evidence**, and the resulting **knowledge state**.

```text
Observation
     │
     ▼
Claim
     │
     ▼
Evidence
     │
     ▼
Verification
     │
     ▼
Knowledge State
```

An observation is information obtained from a source, experiment, API, sensor, agent or interaction.

A claim is an assertion made by an agent.

Evidence may support or contradict a claim.

A knowledge state represents the current interpretation of the claim together with its history and supporting information.

Therefore:

> **A claim is not automatically knowledge.**

---

# 6. Knowledge Objects

The canonical unit of GCN is the **Knowledge Object**.

Objects may represent:

* claims;
* concepts;
* entities;
* observations;
* relationships;
* procedures;
* evidence;
* hypotheses;
* memory events.

Example:

```json
{
  "id": "kg:7f91...",
  "type": "claim",
  "subject": "Python",
  "predicate": "used_for",
  "object": "machine_learning",
  "author": "did:node:abc123",
  "created": "2026-08-18T00:00:00Z",
  "evidence": ["ev:123", "ev:456"],
  "confidence": 0.98,
  "version": 3,
  "content_hash": "...",
  "signature": "..."
}
```

The schema is intentionally experimental and remains an open research question.

---

# 7. Knowledge State and Events

Knowledge is not treated as a permanently fixed value.

Changes are represented as events:

```text
CREATE
UPDATE
LINK
UNLINK
MERGE
SPLIT
SUPPORT
CONTRADICT
VERIFY
RETRACT
REINFORCE
DECAY
```

Example lifecycle:

```text
Observation A
      │
      ▼
    Claim
      │
      ▼
  Evidence B
      │
      ▼
Confidence ↑
      │
      ▼
  Evidence C
      │
      ▼
Confidence ↓
      │
      ▼
 Retraction
```

This produces:

```text
Immutable History
       +
Mutable Knowledge State
```

Historical states remain reconstructable rather than being silently overwritten.

---

# 8. Contradictory Knowledge

GCN does not require conflicting claims to be automatically deleted.

```text
Claim A
   │
   └── contradicts ──► Claim B
```

Both claims may remain available together with:

* provenance;
* evidence;
* timestamps;
* confidence;
* verification history;
* temporal context.

The consuming AI can then determine which claim is appropriate for a particular context.

This treats disagreement as information rather than corruption.

---

# 9. Memory Model

GCN distinguishes several forms of machine memory.

### Working Memory

Temporary information required for current reasoning.

### Episodic Memory

Records of events and experiences.

### Semantic Memory

Generalized concepts and facts extracted from experience.

### Associative Memory

Relationships between concepts, entities, events and experiences.

### Procedural Memory

Knowledge describing how tasks can be performed.

### Global Knowledge

Information explicitly published for use by other systems.

Local memory does not automatically become global knowledge.

---

# 10. Knowledge Graph

Knowledge Objects can form a graph:

```text
              Python
                │
        ┌───────┼────────┐
        ▼       ▼        ▼
 Programming    AI     Software
        │
        ▼
 Machine Learning
```

Graph traversal can support:

* associative retrieval;
* relationship discovery;
* context expansion;
* knowledge consolidation;
* contradiction detection.

The graph is the structural representation of relationships, not merely a visualization layer.

---

# 11. Semantic and Hybrid Retrieval

Embeddings may be used as retrieval indexes.

They are not the canonical representation of knowledge.

```text
Knowledge Objects
       │
       ▼
   Embeddings
       │
       ▼
  Vector Index
       │
       ▼
Semantic Retrieval
```

GCN combines semantic retrieval with graph and trust signals:

```text
                    Query
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Semantic Search          Graph Search
          │                       │
          └───────────┬───────────┘
                      ▼
                Knowledge Rank
                      │
                      ▼
                Evidence Check
                      │
                      ▼
                Context Builder
                      │
                      ▼
                     AI
```

Ranking may incorporate:

```text
semantic relevance
+ graph relevance
+ provenance
+ evidence
+ verification
+ recency
- contradiction
```

The objective is to retrieve not only relevant information, but information whose origin and state can be evaluated.

---

# 12. Provenance, Integrity and Trust

GCN explicitly separates **integrity** from **truth**.

```text
Integrity ≠ Truth
```

Cryptography can establish:

* who signed an object;
* whether its content changed;
* whether an event is authentic;
* whether a history is consistent;
* whether a Merkle proof is valid.

Cryptography cannot prove that a real-world statement is true.

Truth estimation may instead depend on:

* evidence;
* independent observations;
* reproducibility;
* source quality;
* temporal consistency;
* domain-specific validation;
* contradiction analysis.

---

# 13. Knowledge Passport

A **Knowledge Passport** provides a compact view of an object's provenance and state.

```text
Knowledge Passport

ID:
kg:7f91...

Author:
did:node:abc123

Version:
7

Confidence:
0.98

Evidence:
4

Independent Verification:
3

Contradictions:
0

Content Hash:
...

Verification:
Valid
```

It allows an AI to evaluate both:

> **What is known?**

and:

> **Why should it be trusted?**

---

# 14. Distributed State

GCN is designed for independent nodes.

```text
Node A ───── Node B
   │             │
   │             │
   └──── Node C ─┘
```

Nodes may provide:

* peer discovery;
* authentication;
* synchronization;
* partial replication;
* knowledge exchange;
* state verification.

A node does not need to store the entire global knowledge state.

## Partial Replication

GCN separates:

```text
Global Logical State
```

from:

```text
Local Physical Storage
```

A node may store only:

* local knowledge;
* domain-specific knowledge;
* frequently accessed objects;
* cached objects;
* selected historical states.

---

# 15. Merkle State

Large knowledge states can be represented using Merkle structures.

```text
                 GLOBAL ROOT
                /           \
              H1             H2
             /  \           /  \
           H11  H12       H21  H22
```

Merkle structures can provide:

* state verification;
* partial synchronization;
* compact proofs;
* distributed snapshots.

A node can therefore verify selected objects without possessing the complete global state.

---

# 16. AI Adapter Protocol

The **AI Adapter Protocol** is the model-independent interface between an AI system and GCN.

Conceptual operations include:

```text
connect()
query()
retrieve()
publish()
verify()
explain()
relate()
subscribe()
synchronize()
```

### Query

Find relevant knowledge.

### Retrieve

Return knowledge objects and evidence.

### Publish

Publish observations, claims or other objects.

### Verify

Validate integrity and provenance.

### Explain

Return origin and history.

### Relate

Explore graph relationships.

### Synchronize

Synchronize selected knowledge with a local node.

The protocol is independent of the underlying AI architecture.

---

# 17. Privacy and Security

Global knowledge does not imply global publication.

GCN distinguishes:

```text
PUBLIC KNOWLEDGE
PRIVATE MEMORY
PERSONAL DATA
ENCRYPTED DATA
```

Potential privacy mechanisms include:

* encryption;
* access control;
* selective disclosure;
* private knowledge domains;
* zero-knowledge proofs.

Security threats include:

* false information;
* memory poisoning;
* malicious agents;
* compromised keys;
* replay attacks;
* spam;
* Sybil attacks;
* outdated knowledge;
* network partitions.

Potential defenses include:

* digital signatures;
* content hashes;
* provenance;
* rate limiting;
* access control;
* reputation;
* anomaly detection;
* independent verification.

The number of identities must not be treated as evidence of truth.

```text
Number of identities
        ≠
Independent evidence
```

---

# 18. Blockchain

Blockchain is optional.

It may be used for:

* identity anchoring;
* timestamps;
* state roots;
* event ordering;
* authorization;
* public proofs.

Knowledge itself can remain in distributed storage.

```text
GCN
 │
 ├── P2P
 ├── Knowledge Graph
 ├── Distributed Storage
 ├── Vector Index
 ├── Merkle State
 │
 └── Blockchain
        optional
```

Therefore:

> **Blockchain is an implementation option for the trust layer, not the definition of GCN.**

Blockchain cannot:

* make information true;
* create intelligence;
* replace AI;
* replace distributed storage;
* solve knowledge quality.

---

# 19. Positioning

## GCN vs RAG

Traditional RAG:

```text
Documents
   ↓
Embeddings
   ↓
Vector Search
   ↓
Context
   ↓
LLM
```

GCN:

```text
Observations
   ↓
Claims
   ↓
Evidence
   ↓
Knowledge Objects
   +
Graph
   +
Vector Index
   +
Provenance
   +
Version History
   ↓
Knowledge Ranking
   ↓
AI
```

GCN therefore treats retrieved information as structured, persistent and provenance-aware knowledge rather than simply retrieved text.

## GCN vs Database

A database primarily represents stored data.

GCN additionally represents:

* origin;
* evidence;
* relationships;
* contradictions;
* versions;
* verification history.

Its purpose is not to replace databases, but to define a machine-oriented knowledge layer above or across storage systems.

---

# 20. Knowledge Persistence

The central persistence experiment is:

```text
AI A
 │
 └── discovers X
          │
          ▼
         GCN
          │
     AI A removed
          │
          ▼
         AI B
          │
          ▼
      retrieves X
          │
          ▼
        verifies
```

If AI B can retrieve and use X without access to AI A's:

* model;
* memory;
* hardware;
* execution environment,

then the experiment demonstrates that the knowledge itself persisted independently of the original model.

---

# 21. Minimal Multi-Agent Experiment

A stronger experiment introduces independent verification.

```text
AI A
 │
 └── discovers X
       │
       ▼
      GCN
       │
 ┌─────┼─────┐
 ▼     ▼     ▼
AI B  AI C  AI D
 │     │     │
verify test  retrieve
```

Possible result:

```text
A → Claim X
B → supports X
C → supports X
D → contradicts X
```

The system should preserve the competing evidence and produce an updated knowledge state rather than silently selecting one claim.

---

# 22. Evaluation

GCN should be evaluated experimentally.

### Knowledge Transfer

Can AI B retrieve and use knowledge created by AI A?

### Model Independence

Does knowledge remain usable after replacing the original model?

### Verification

Can nodes detect:

* modified objects;
* invalid signatures;
* broken event history;
* invalid Merkle proofs?

### Conflict Representation

Can contradictory claims coexist without loss of provenance?

### Retrieval Quality

Does:

```text
Graph + Vector + Provenance
```

outperform:

```text
Vector Search
```

alone?

### Synchronization Cost

Measure:

* bandwidth;
* storage;
* CPU;
* latency.

### Poisoning Resistance

Measure how easily malicious knowledge can influence retrieval and downstream reasoning.

---

# 23. Current Prototype

`BlockcoinWitres` is an experimental implementation exploring selected GCN concepts.

Current work includes experimental components for:

* AI memory;
* associative memory graphs;
* structured knowledge relationships;
* local retrieval;
* persistent machine memory;
* AI integration.

The following remain research targets:

* standardized GCN protocol;
* interoperable AI Adapter;
* multi-node synchronization;
* cryptographic identity;
* distributed state;
* Merkle synchronization;
* large-scale P2P operation;
* federation.

This document describes the target architecture, not a claim that all components are already implemented.

---

# 24. Roadmap

## Phase 1 — Local Knowledge

Implement:

* Knowledge Objects;
* Claims;
* Evidence;
* Memory Graph;
* local retrieval.

**Goal:** persistent structured machine memory.

## Phase 2 — Cryptographic Knowledge

Implement:

* content hashes;
* signed events;
* object identity;
* version history;
* Merkle roots.

**Goal:** verifiable knowledge history.

## Phase 3 — Two Nodes

```text
Node A ↔ Node B
```

Implement:

* P2P communication;
* synchronization;
* signatures;
* verification.

**Goal:** Node B can receive and verify knowledge created by Node A.

## Phase 4 — Independent AI

Connect heterogeneous AI systems through the same adapter.

**Goal:** knowledge transfer without sharing the original model or memory.

## Phase 5 — Knowledge Evolution

Implement:

* support;
* contradiction;
* verification;
* retraction;
* confidence updates.

**Goal:** preserve and evolve competing knowledge states.

## Phase 6 — Distributed State

Implement:

* partial replication;
* Merkle synchronization;
* node discovery;
* state proofs.

**Goal:** verifiable distributed knowledge without full replication.

## Phase 7 — Multi-Agent Network

Connect larger numbers of independent AI systems.

**Goal:** measure whether shared persistent knowledge provides measurable advantages over isolated systems.

---

# 25. Open Research Questions

1. What should the canonical Knowledge Object format be?
2. How should evidence be represented?
3. How should confidence be calculated?
4. How should contradictory claims be ranked?
5. How should source reliability be measured?
6. How can false information be detected?
7. How can Sybil attacks be resisted?
8. How should private memory interact with public knowledge?
9. How should partial graphs synchronize?
10. How should semantic indexes be distributed?
11. How should knowledge be ranked?
12. When is blockchain actually useful?
13. When are Merkle structures sufficient?
14. How should AI Adapter compatibility be standardized?
15. Can knowledge survive model replacement?
16. Can independently operated AI systems contribute to the same evolving knowledge state?
17. Does graph-based shared memory improve reasoning?
18. Does collective machine experience provide measurable advantages over isolated AI systems?

---

# 26. Non-Goals

GCN is not intended to be:

* an AGI;
* a replacement for neural networks;
* a blockchain-based AI;
* a universal truth oracle;
* a machine consciousness system.

It is an experimental architecture for persistent, interoperable and verifiable machine knowledge.

---

# 27. Conclusion

GCN proposes a separation between intelligence and persistent machine knowledge.

```text
Model       → intelligence
Node        → computation
Memory      → experience
Knowledge   → structured information
Graph       → relationships
Evidence    → support
Provenance  → history
Cryptography→ integrity
Protocol    → interoperability
```

The core hypothesis is that knowledge produced by one AI may remain available, verifiable and usable by another AI without transferring the original model or execution environment.

The fundamental experiment is therefore:

```text
AI A
 ↓
learns X
 ↓
publishes X
 ↓
AI B verifies X
 ↓
AI C retrieves X
 ↓
AI C builds upon X
```

If this can be demonstrated reliably, it would provide experimental evidence that machine knowledge can persist and move between independent AI systems.

GCN therefore does not attempt to create intelligence by itself.

It attempts to investigate whether **persistent shared machine knowledge can become infrastructure for heterogeneous AI systems**.

> **Intelligence may remain distributed while knowledge becomes interoperable.**

**GCN — Research Concept**
**BlockcoinWitres — Experimental Implementation**
