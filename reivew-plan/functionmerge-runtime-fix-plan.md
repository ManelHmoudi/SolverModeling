# FunctionMerge Runtime Fix Plan

## Summary

The FunctionMerge failure is not caused by the HTML report. The report is never reached because CPLEX runs until the configured `420s` time limit and ends with `solutions = 0`, meaning it did not find a feasible incumbent solution for the 20-client instance.

The 20-client data is not obviously invalid from the basic capacity checks. The main issue is that increasing the client count makes the exact mixed-integer model much larger and harder to search.

## Diagnosis

- The 20-client instance has `21` nodes, `5` periods, and `5` vehicles.
- It creates `420` possible directed arcs per period/vehicle.
- Before presolve, the route decision variables alone create `10,500` binary variables.
- After presolve, CPLEX still reports a reduced MIP with `4,958` binary variables.
- The console log shows CPLEX spends the full `420.11s` in branch-and-cut and still has `solutions = 0`.
- This means the app error is a solver/model runtime issue, not a browser or HTML-generation issue.

## Main Fixes

1. Add two FunctionMerge execution modes:
   - `fast`: default mode for real-life quick report generation.
   - `exact`: CPLEX optimization mode for research/comparison.

2. Make fast mode the default web path:
   - `/function-merge/run` should generate a practical report quickly.
   - The fast result does not need to prove mathematical optimality.

3. Keep exact CPLEX available explicitly:
   - Use `/function-merge/run?mode=exact`.
   - Keep `timelimit` support for exact mode, for example `/function-merge/run?mode=exact&timelimit=3600`.

4. Repair the timing formulation in exact mode:
   - Replace shared `tau[i,t]` arrival time with vehicle-specific `tau[i,t,k]`.
   - Replace shared `tau_return[t]` coupling with vehicle-specific return times.
   - Keep a period-level return-time summary for the HTML report.

5. Add fallback behavior:
   - If exact CPLEX times out without a feasible solution, return a fast heuristic report instead of showing only an error page.
   - Include a warning in the report metadata or web response so the user knows the fallback was used.

## Test Checklist

### 20-Client Instance

- Run fast mode on `Project_Irp/data/instance_20_clients.json`.
- Confirm the HTML report is generated quickly.
- Confirm every client-period demand is delivered.
- Confirm vehicle capacities are respected.
- Confirm refrigerated demand uses refrigerated vehicles.
- Confirm non-refrigerated demand uses non-refrigerated vehicles.
- Confirm route return times stay within `tau_max`.

### 5-Client Instance

- Run exact mode on `Project_Irp/data/instance_5_clients.json`.
- Confirm CPLEX still solves successfully.
- Confirm the report schema remains compatible with the existing HTML renderer.

### 3-Client Instance

- Run exact mode on `Project_Irp/data/instance_3_clients.json`.
- Confirm the small baseline instance still solves.
- Confirm objective values and route output are still produced.

### Code Health

- Run Python syntax checks on edited modules.
- Check that no Python behavior is changed by this review-plan document.

## Assumptions

- The folder name must be `reivew-plan` exactly, even though it appears to be a typo.
- This document is only for review and planning.
- No Python implementation changes are included in this step.
