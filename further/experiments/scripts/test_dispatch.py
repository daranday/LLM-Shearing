from dispatch import DummyConfig, dummy, launch_workflow


def test_dummy():
    launch_workflow(dummy, group="test", cpu=True)(DummyConfig())
