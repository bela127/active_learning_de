import pytest
from dataclasses import dataclass
from active_learning_de.new_api.core.configuration import Configurable

@dataclass
class Test(Configurable):
    x: int
    y: int = 5

    def __init__(self, x: int, y: int = 5) -> None:
        print("init")
        self.x = x
        self.y = y
        super().__init__()

    def __str__(self) -> str:
        return f"x = {self.x}, i = {self.y}"




def test_configurable_fail():
    test_config: Configurable = Test(0, 6)

    with pytest.raises(AttributeError):
        assert test_config.x == 0


def test_configurable_one():
    test_config: Configurable = Test(0, 6)
    test: Test = test_config()

    assert test.x == 0
    assert test.y == 6

def test_configurable_change():
    test_config: Configurable = Test(0, 6)
    test: Test = test_config()

    assert test.x == 0
    assert test.y == 6

    test.x = 3
    test.y = 4

    assert test.x == 3
    assert test.y == 4

    test: Test = test_config()

    assert test.x == 0
    assert test.y == 6