"""Pipeline exports; load hardware-dependent pipelines only when requested."""
from importlib import import_module

_MODULES = {
    "AssemblyPlan": "assembly_plan",
    "AssemblyStep": "assembly_plan",
    "LongHorizonPipeline": "long_horizon_pipeline",
    "LongHorizonPipelineConfig": "long_horizon_pipeline_config",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
