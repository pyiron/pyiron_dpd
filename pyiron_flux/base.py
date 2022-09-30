from typing import Optional, Tuple, Iterable, Union, Mapping

from pyiron_base import Project
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage


class FlexibleProperty:

    def __init__(self, name, storage_attribute='input'):
        self._name = name
        self._storage_attribute = storage_attribute

    @property
    def type(self):
        return object

    @property
    def default(self):
        return None

    def wrap(self, value):
        '''Called on setting a value before saving it to storage.'''
        return value

    def unwrap(self, value):
        '''Called on getting a value after getting it from storage before returing it.'''
        return value

    def get_storage(self, instance):
        return getattr(instance, self._storage_attribute)

    def __get__(self, instance, owner):
        try:
            value = self.get_storage(instance)[self._name]
        except KeyError:
            value = self.default
            self.get_storage(instance)[self._name] = value

        if value is None:
            raise AttributeError(f"property {self._name} not yet set!")

        return self.unwrap(value)

    def setter(self, storage, name, value):
        storage[name] = value

    def __set__(self, instance, value):
        if not isinstance(value, self.type):
            raise TypeError(f"value must be of type {self._type}, not {value}!")
        self.setter(self.get_storage(instance), self._name, self.wrap(value))
        instance.sync()

class ScalarProperty(FlexibleProperty):
    def __init__(self, name, type=object, default=None, storage_attribute='input'):
        super().__init__(name=name, storage_attribute=storage_attribute)
        self._type = type
        self._default = default

    @property
    def type(self):
        return self._type

    @property
    def default(self):
        return self._default

class StructureProperty(FlexibleProperty):

    @property
    def type(self):
        return Iterable, Mapping, StructureStorage

    @property
    def default(self):
        return StructureStorage()

    def setter(self, storage, name, value):
        if isinstance(value, Mapping):
            if 'name' in storage: del storage[name]
            s = self.default
            for sname, structure in value.items():
                if not isinstance(structure, Atoms):
                    raise TypeError(f"value must be of type Atoms, not {type(structure)}");
                s.add_structure(structure, identifier=sname)
            storage[name] = s
        elif isinstance(value, Iterable):
            if 'name' in storage: del storage[name]
            s = self.default
            for structure in value:
                if not isinstance(structure, Atoms):
                    raise TypeError(f"value must be of type Atoms, not {type(structure)}");
                s.add_structure(structure)
            storage[name] = s
        else:
            storage[name] = value

class IterableProperty(FlexibleProperty):

    def __init__(self, name, type=None, storage_attribute='input'):
        super().__init__(name=name, storage_attribute=storage_attribute)
        self._type = type

    @property
    def type(self):
        return Iterable

    @property
    def default(self):
        return []

    def wrap(self, value):
        return list(value)

    def unwrap(self, value):
        return tuple(value)

class WorkFlow:

    def __init__(self, project: Project, name: Optional[str] = None):
        if name is None:
            try:
                name = self._default_name
            except AttributeError:
                raise ValueError('Either name must be passed or workflow must define a _default_name!')
        self._project = project.create_group(name)
        try:
            self._project.data.read()
        except KeyError:
            pass

    @property
    def project(self):
        return self._project

    @property
    def input(self):
        if 'input' not in self.project.data:
            self.project.data.create_group('input')
        return self.project.data.input

    @property
    def output(self):
        if 'output' not in self.project.data:
            self.project.data.create_group('output')
        return self.project.data.output

    def sync(self):
        self.project.data.write()

    def __del__(self):
        self.sync()
