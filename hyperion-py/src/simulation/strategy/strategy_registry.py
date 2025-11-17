from typing import Type, Dict
from .strategy import Strategy


class StrategyRegistry:
    _registry: Dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[Strategy]):
        name = name.lower()
        if name in cls._registry:
            raise ValueError(f"Strategy '{name}' already registered.")
        cls._registry[name] = strategy_cls

    @classmethod
    def get(cls, name: str) -> Type[Strategy]:
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(
                f"Strategy '{name}' not registered. " f"Registered strategies: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def create(cls, name: str, simulator, capital, **kwargs) -> Strategy:
        strategy_cls = cls.get(name)
        return strategy_cls(simulator=simulator, capital=capital, **kwargs)

    @classmethod
    def list(cls):
        return list(cls._registry.keys())


# Decorator for adding strategies to the registry
def register_strategy(name: str):
    def decorator(cls):
        StrategyRegistry.register(name, cls)
        return cls

    return decorator
