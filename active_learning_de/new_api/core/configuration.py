from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Type, Dict, Tuple
    from typing_extensions import Self #type: ignore


class ConfigurableMeta(type):

    def __call__(cls: type, *args: Any, **kwargs: Any) -> Any:
        obj: Configurable = cls.__new__(cls, *args, **kwargs)
        return obj

class Configurable(metaclass = ConfigurableMeta):
    __cls: Type = type
    __args: Tuple[Any,...] = ()
    __kwargs: Dict[str, Any] = {}
    __initialized: bool = False

    def __getattribute__(self, __name: str) -> Any:
        initialized = super().__getattribute__("_Configurable__initialized")
        if not initialized:
            try:
                attr_result = super().__getattribute__(__name)
            except AttributeError:
                raise AttributeError("Configurable has not been initialized")
        else:
            attr_result = super().__getattribute__(__name)
        return attr_result


    def __new__(cls: Type[Self], *args, **kwargs) -> Self:
        obj: Self = super(Configurable, cls).__new__(cls)
        obj.__cls = cls
        obj.__args = args
        obj.__kwargs = kwargs
        return obj
    
    def __call__(self) -> Self:
        self.__initialized = True
        self.__cls.__init__(self, *self.__args, **self.__kwargs)
        return self
