import pytest
from dataclasses import dataclass
from ide.core.configuration import Configurable, InitError

@dataclass
class Test(Configurable):
    x: int
    y: int = 5

def test_configurable_print():
    test_config: Configurable = Test(0, 6)
    assert test_config.__repr__() == "Test(0, 6)"


def test_configurable_fail():
    test_config: Configurable = Test(0, 6)

    with pytest.raises(InitError) as exc_info:
        assert test_config.x == 0 #Test not initialized
    assert exc_info.type is InitError
    
    test = test_config()

    with pytest.raises(AttributeError) as exc_info:
        assert test.z == 0
    assert exc_info.type is AttributeError


def test_configurable_one():
    test_config: Configurable = Test(0, 6)
    test: Test = test_config()

    assert test.x == 0
    assert test.y == 6

def test_configurable_change():
    test_config: Configurable = Test(0, 6)
    test1: Test = test_config() #init

    assert test1.x == 0 #test init
    assert test1.y == 6

    test1.x = 3 #change something
    test1.y = 4

    assert test1.x == 3 #test change
    assert test1.y == 4

    test2: Test = test_config() #re-init

    assert test2.x == 0 #test original init
    assert test2.y == 6

    assert test1.x == 3 #test not re-init
    assert test1.y == 4