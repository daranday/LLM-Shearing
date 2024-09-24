from dispatch import DummyConfig, dummy, launch_workflow

if __name__ == "__main__":
    launch_workflow(dummy, group="test", cpu=True)(DummyConfig())
