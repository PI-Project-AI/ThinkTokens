# V11: Moderate Scale-Up of Air-Gap VQ Transformer

**Status:** Ready to Run
**Goal:** Validate "Air-Gap" architecture on harder tasks (3-digit mixed arithmetic).

## Configuration
*   **Architecture:**
    *   Reasoner: 6 Layers, 8 Heads, 384 Dim
    *   Speaker: 6 Layers, 8 Heads, 384 Dim
    *   VQ: 1024 Codes, 384 Dim
    *   Context: 32 IR tokens
*   **Data:**
    *   3-digit Addition/Subtraction
    *   2-digit Multiplication (Max complexity 2 for '*')
    *   200k Training Samples
*   **Training:**
    *   Batch Size: 128
    *   Epochs: 40
    *   Loss: CE + VQ + 0.1 * Entropy

## Usage
Run `./run_v11.sh`
