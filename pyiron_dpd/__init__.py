from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .base import (
        FlexibleProperty,
        ScalarProperty,
        StructureProperty,
        IterableProperty,
        WorkFlow
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
