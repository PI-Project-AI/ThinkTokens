# Two-Model Outperform Signal (Current)

## Question

Can the two-model LA architecture outperform a classic CoT-style approach?

## Current Evidence Level

### Supported now (local proxy level)

1. **LA > matched text-only proxy (weaker baseline family)**
   - TM-EXP-04 IID random: strong gain.
   - TM-EXP-04 OOD `*->+`: positive gain with coverage-adjusted test set.

2. **Causal dependence is present on LA side**
   - Shuffle/drop controls degrade LA strongly in successful runs.

3. **Breadth is partial, not universal (TM-EXP-05 sweep)**
   - TM-EXP-05 sweep: 5/9 op-pair holdouts green.
   - All 9 pairs have positive LA-text gain, but 4/9 fail full criteria.

4. **Some weak pairs are optimization-limited**
   - TM-EXP-06 on `plus->mul`: large gain jump under stronger budget.
   - `minus->minus` improves but remains below causal thresholds.

5. **Against stronger text-AR baseline, story is mixed**
   - TM-EXP-07: original LA loses strongly on hard pairs.
   - TM-EXP-08: factorized LA recovers; wins on `+->*`, near-parity on `-->-`.
   - TM-EXP-09: subtraction-focused curriculum does not improve `-->-` beyond TM-EXP-08 near-parity.
   - TM-EXP-10: scheduled-sampling objective change also fails to improve `-->-` margin.

### Not supported yet (final claim level)

- No benchmark against external classic CoT LLM(s) yet.
- No broad natural-language reasoning benchmark suite yet.
- Current tasks are synthetic/local.

## Practical Status

- **Continue investing** in two-model LA line: yes.
- **Claim “better than classic CoT LLM” publicly**: still no.

## Suggested Gate to Next Claim

Before final outperform claim, require:
1. Matched baseline upgrade (stronger CoT-style local generator baseline).
2. Coverage-clean OOD suite with multiple task families.
3. Stable multi-seed win + causal controls.
4. External benchmark comparison protocol (frozen setup).
