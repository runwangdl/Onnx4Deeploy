import sys
import os
import subprocess

OPERATORS_DIR = "./Tests/Operators/"

def list_operators():
    """List all available operators in the Tests/Operators/ directory."""
    if not os.path.exists(OPERATORS_DIR):
        print("⚠️ Operators directory does not exist!")
        return []

    operators = [
        d for d in os.listdir(OPERATORS_DIR)
        if os.path.isdir(os.path.join(OPERATORS_DIR, d)) and
        os.path.exists(os.path.join(OPERATORS_DIR, d, "testgenerate.py"))
    ]

    return operators

def print_help():
    """Print usage information and available operators."""
    print("\nUsage: python Onnx4Deeploy.py <operator_name> [save_path]")
    print("Runs the corresponding testgenerate.py for a given operator.\n")
    print("Arguments:")
    print("  <operator_name>    Name of the operator to run.")
    print("  [save_path]        (Optional) Custom path to save ONNX models and outputs.\n")
    print("Available operators:")
    
    operators = list_operators()
    if operators:
        for op in operators:
            print(f"  - {op}")
    else:
        print("  No operators found in ./Tests/Operators/\n")

def run_operator(operator_name, save_path=None):
    """Run the corresponding testgenerate.py for a given operator with an optional save path."""
    
    operator_script = os.path.join(OPERATORS_DIR, operator_name, "testgenerate.py")

    if not os.path.exists(operator_script):
        print(f"❌ Error: Operator '{operator_name}' does not exist!")
        sys.exit(1)

    if save_path:
        subprocess.run(["python", operator_script, save_path])
    else:
        subprocess.run(["python", operator_script])

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print_help()
        sys.exit(0)

    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("❌ Invalid arguments! Use --help for usage instructions.")
        sys.exit(1)

    operator_name = sys.argv[1]
    save_path = sys.argv[2] if len(sys.argv) == 3 else None 

    run_operator(operator_name, save_path)
