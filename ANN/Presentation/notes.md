# Paper: Lightweight and Fast Time-Series Anomaly Detection via Point-Level and Sequence-Level Reconstruction Discrepancy

### 3 existing anomaly-scoring approaches discussed in the paper.
- 


### The two types of anomalies the paper targets
- **Explicit Outliers**
    - An explicit outlier has a pattern that clearly deviates from normal observations.
    - For example:
        - Normal: 10, 11, 10, 12, 11, 10
        - Anomaly: 10, 11, 45, 12, 11, 10
    - This is detected using **Point-level branch**.
- **Implicit Outliers**
    - An implicit outlier is more difficult. Some anomalous timestamps may look individually normal, and their local relationships may resemble those of normal timestamps.
    - For example, imagine a machine that normally follows a repeating cycle:
        - Normal cycle: A → B → C → D
    - An abnormal event might produce:
        - Observed cycle: A → C → B → D
    - The individual readings may all lie within normal numerical ranges. However, the order or temporal relationships are unusual.
    - This is detected using **Sequence-level branch**.

### The 2 branches
**Point-level branch**

Examines an individual timestamp and how it relates to its local and global neighbours. Particularly useful for explicit anomalies that have a distinctive point-level pattern.

**Sequence-level branch**

Examines a short subsequence instead of an isolated reading. It can detect anomalies whose individual timestamps appear less unusual but whose sequence-level relationships differ.

*The purpose of having both branches is to reduce the weaknesses of either branch operating alone.*

### What exactly is LFTSAD?
LFTSAD stands for Lightweight and Fast Time-Series Anomaly Detection.

It consists of three major components:
1. Point-level reconstruction-discrepancy learning.
2. Sequence-level reconstruction-discrepancy learning.
3. Reconstruction-discrepancy-based anomaly scoring.