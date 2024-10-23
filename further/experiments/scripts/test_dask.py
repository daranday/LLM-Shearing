import dask
from dask.distributed import Client


def inc(x):
    print(f"inc({x})")
    return x + 1


def double(x):
    print(f"double({x})")
    return x * 2


def add(x, y):
    print(f"add({x}, {y})")
    return x + y


if __name__ == "__main__":
    client = Client()
    print(client.dashboard_link)

    data = [1, 2, 3, 4, 5]
    output = []
    for x in data:
        a = dask.delayed(inc)(x)
        b = dask.delayed(double)(x)
        c = dask.delayed(lambda a, b: b)(a, b)
        output.append(c)

    total = dask.delayed(sum)(output)
    print("Total:", total.compute())

    import time

    time.sleep(1000)
