from __future__ import annotations
from typing import TYPE_CHECKING, Protocol

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from typing import Any, Type, Dict, Tuple
    from typing_extensions import Self


class ConfigurableMeta(type):

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = cls.__new__(cls, *args, **kwargs)
        return obj

class Configurable(metaclass = ConfigurableMeta):
    __cls: Type = type
    __args: Tuple[Any,...] = ()
    __kwargs: Dict[str, Any] = {}
    __initialized: bool = False

    def __getattribute__(self, __name: str) -> Any:
        print("get: ",__name)
        initialized = super().__getattribute__("_Configurable__initialized")
        if not initialized:
            try:
                attr_result = super().__getattribute__(__name)
            except AttributeError:
                raise AttributeError("Configurable has not been initialized")
        else:
            attr_result = super().__getattribute__(__name)
        print("geting: ",attr_result)
        return attr_result


    def __new__(cls: Type[Self], *args, **kwargs) -> Self:
        print("new")
        obj: Self = super(Configurable, cls).__new__(cls)
        obj.__cls = cls
        obj.__args = args
        obj.__kwargs = kwargs
        return obj# Config(cls, args, kwargs)
    
    def __call__(self) -> Self:
        print("call")

        self.__initialized = True
        self.__cls.__init__(self, *self.__args, **self.__kwargs)
        return self


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

test_config: Configurable = Test(0, 6)
#print(test_config)
test: Test = test_config()

print(test)