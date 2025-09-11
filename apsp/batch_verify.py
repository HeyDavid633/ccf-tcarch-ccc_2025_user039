import sys
import os
import subprocess

RTOL = 1e-5
ATOL = 1e-6

def is_close(a, b, abs_tol=ATOL, rel_tol=RTOL):
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))

def compare_output(student_output, reference_output):
    try:
        tokens1 = student_output.strip().split()
        tokens2 = reference_output.strip().split()

        if len(tokens1) != len(tokens2):
            print(f"FAIL: Different number of elements. Expected {len(tokens2)}, got {len(tokens1)}")
            return False

        for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
            try:
                v1 = float(t1)
                v2 = float(t2)
                if not is_close(v1, v2):
                    print(f"FAIL: Mismatch at Element {i+1}.")
                    print(f"      Expected: {t2}")
                    print(f"      Got:      {t1}")
                    return False
            except ValueError:
                if t1 != t2:
                    print(f"FAIL: Mismatch at Element {i+1}.")
                    print(f"      Expected: {t2}")
                    print(f"      Got:      {t1}")
                    return False
        
        return True

    except Exception as e:
        print(f"Error during comparison: {e}")
        return False

def run_test(program_path, input_file, expected_output_file):
    """
    运行单个测试用例
    """
    try:
        # 运行程序
        result = subprocess.run([program_path, input_file], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"ERROR: Program failed with return code {result.returncode}")
            print(f"Stderr: {result.stderr}")
            return False
        
        # 读取期望输出
        with open(expected_output_file, 'r') as f:
            expected_output = f.read().strip()
        
        # 比较输出
        student_output = result.stdout.strip()
        return compare_output(student_output, expected_output)
        
    except subprocess.TimeoutExpired:
        print("ERROR: Program timed out")
        return False
    except FileNotFoundError as e:
        print(f"ERROR: File not found: {e}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def batch_test(test_dir, program_path):
    """
    批量运行所有测试用例
    """
    # 获取所有测试文件
    test_files = []
    for i in range(1, 11):  # 假设有10个测试用例
        input_file = os.path.join(test_dir, f"{i}.in")
        output_file = os.path.join(test_dir, f"{i}.out")
        
        if os.path.exists(input_file) and os.path.exists(output_file):
            test_files.append((input_file, output_file))
    
    if not test_files:
        print("No test files found!")
        return
    
    print(f"Found {len(test_files)} test cases")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for i, (input_file, output_file) in enumerate(test_files, 1):
        print(f"Running test case {i}: {os.path.basename(input_file)}")
        
        success = run_test(program_path, input_file, output_file)
        
        if success:
            print(f"Test case {i}: PASS")
            passed += 1
        else:
            print(f"Test case {i}: FAIL")
            failed += 1
        
        # print("-" * 40)
    
    print("=" * 60)
    print(f"Summary: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("All tests passed! ✅")
        return True
    else:
        print("Some tests failed! ❌")
        return False

if __name__ == "__main__":
    DEFAULT_TEST_DIR = "testcases/"
    DEFAULT_PROGRAM_PATH = "./apsp"
    test_dir = DEFAULT_TEST_DIR
    program_path = DEFAULT_PROGRAM_PATH
    
    success = batch_test(test_dir, program_path)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)