# PoRT: Robust LLM Unlearning via Post Judgment and Multi-Round Thinking

Official implementation for the paper **"Robust LLM Unlearning via Post Judgment and Multi-Round Thinking"**. PoRT is a new framework for robust LLM unlearning that withstands adversarial attacks where existing pre-filtering methods fail.
<img width="3217" height="1327" alt="PoRT" src="./resource/PoRT.png" />
The core pipeline of PoRT, which uses a post-judgment and self-correction mechanism to handle unsafe or ambiguous outputs.


## Core Contributions

*   **A New Robust Unlearning Framework (PoRT):** We introduce a novel, three-stage pipeline (IPC, Post-Judgment, SMT) that moves beyond vulnerable pre-filtering to a more robust post-judgment paradigm.
*   **Systematic Robustness Analysis:** We are the first to systematically evaluate the vulnerability of pre-filtering unlearning methods to tailored adversarial attacks (Prefix and Composite Attacks).
*   **State-of-the-Art Performance:** PoRT achieves SOTA robustness and unlearning effectiveness on the TOFU and WMDP benchmarks, without compromising model utility or efficiency.
