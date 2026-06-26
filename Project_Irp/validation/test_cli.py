import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_validate_3obj_smoke():
    """validate() with n_obj=3 and 1 run should produce _M3 CSV files."""
    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Remove M3 CSVs if they exist from a previous run
    for f in ['igd_DTLZ1_M3.csv', 'igd_DTLZ2_M3.csv',
              'igd_DTLZ3_M3.csv', 'igd_DTLZ4_M3.csv', 'summary_M3.csv']:
        path = os.path.join(results_dir, f)
        if os.path.exists(path):
            os.remove(path)

    from validation.main_validation import validate
    validate(n_runs=1, n_obj=3)

    # Check that M3 files were created
    for name in ['DTLZ1', 'DTLZ2', 'DTLZ3', 'DTLZ4']:
        path = os.path.join(results_dir, f'igd_{name}_M3.csv')
        assert os.path.exists(path), f"Missing: {path}"
        with open(path) as f:
            header = f.readline().strip()
        assert header == 'run,seed,igd', f"Wrong header in {path}: {header}"

    summary = os.path.join(results_dir, 'summary_M3.csv')
    assert os.path.exists(summary), f"Missing: {summary}"
    with open(summary) as f:
        lines = f.readlines()
    assert lines[0].strip() == 'Problem,IGD_Best,IGD_Median,IGD_Worst'
    assert len(lines) == 5, f"Expected 5 lines (header + 4 problems), got {len(lines)}"

    print("test_validate_3obj_smoke passed.")

def test_validate_4obj_still_works():
    """validate() with n_obj=4 should produce _M4 CSV files."""
    from validation.main_validation import validate
    validate(n_runs=1, n_obj=4)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    summary = os.path.join(results_dir, 'summary_M4.csv')
    assert os.path.exists(summary), f"Missing: {summary}"
    print("test_validate_4obj_still_works passed.")

if __name__ == "__main__":
    print("Smoke test n_obj=3 (1 run x 4 problems)...")
    test_validate_3obj_smoke()
    print("Smoke test n_obj=4 (1 run x 4 problems)...")
    test_validate_4obj_still_works()
    print("All CLI tests passed.")
