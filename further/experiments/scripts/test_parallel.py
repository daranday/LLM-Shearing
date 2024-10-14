import time

import dask
import dask.base
from dask.delayed import delayed
from joblib import Memory

# Set up joblib memory cache
memory = Memory(location=".cache", verbose=0)


# Define tasks


def task_a(x):
    print(f"Running task A with input {x}")
    time.sleep(1)  # Simulate some work
    return x * 2


def task_b(x):
    print(f"Running task B with input {x}")
    time.sleep(1)  # Simulate some work
    return x + 1


def task_c(x):
    print(f"Running task C with input {x}")
    time.sleep(1)  # Simulate some work
    return x * 3


def task_d(x):
    print(f"Running task D with input {x}")
    time.sleep(1)  # Simulate some work
    return x - 1


# Function to check if a task will run or use cached result
def will_run(func, *args, **kwargs):
    try:
        if hasattr(memory, "check_call_in_cache"):
            return not memory.check_call_in_cache(func, args, kwargs)
        memory_info = memory.call_and_shelve(func, *args, **kwargs)
        already_cached = memory_info.get_output() is not None
        if not already_cached:
            memory_info.clear()
        return not already_cached
    except Exception:
        return True


# Create the DAG using dask.delayed
def create_dag(input_value):
    a = delayed(task_a)(input_value)
    b = delayed(task_b)(input_value)
    c = delayed(task_c)(input_value)
    d = delayed(task_d)(input_value)

    # Establish dependencies
    b.compute_as_if_collection = dask.base.optimize(dask.delayed(lambda x, y: y)(b, a))
    c.compute_as_if_collection = dask.base.optimize(dask.delayed(lambda x, y: y)(c, a))
    d.compute_as_if_collection = dask.base.optimize(dask.delayed(lambda x, y: y)(d, c))

    return a, b, c, d


# Main execution function
def run_workflow(input_value):
    print("Task execution preview:")
    print(f"Task A will run: {will_run(task_a, input_value)}")
    print(f"Task B will run: {will_run(task_b, input_value)}")
    print(f"Task C will run: {will_run(task_c, input_value)}")
    print(f"Task D will run: {will_run(task_d, input_value)}")
    print("\nExecuting tasks:")

    a, b, c, d = create_dag(input_value)
    results = dask.compute(a, b, c, d)
    return results


# Example usage
if __name__ == "__main__":
    input_value = 5
    results = run_workflow(input_value)
    print(f"\nResults: {results}")

    print("\nRunning again with the same input:")
    results = run_workflow(input_value)
    print(f"\nResults: {results}")
