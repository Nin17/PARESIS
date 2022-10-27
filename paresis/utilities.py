"""_summary_
"""
import json
import inspect
import warnings
import numpy as np
# TODO change it to private methods
# TODO where is name == hola coming from
# TODO getting key errors for _ things


class Utility:
    """_summary_
    """

    def __init__(self, name, json_file: str) -> None:
        self.name = name
        self.json_file = json_file

    def __repr__(self) -> str:
        """
        String representation of the class

        Returns
        -------
        str
            The string representation of the class object
        """
        if self.__dict__.keys():
            i = max(map(len, list(self.__dict__.keys()))) + 1
            return '\n'.join([f'{k.rjust(i)}: {repr(v)}'
                              for k, v in sorted(self.__dict__.items())])
        return ''

    # TODO change to dictionary and give warning that key will be overwritten if already there
    # TODO dependent on which subclass calls it
    # TODO sort out protected attributes
    # tODO testing on classes with lots of inheritance
    # TODO will defo have to change this when with inheritance i reckon
    def save(self, json_file: str = None) -> None:
        """_summary_

        Parameters
        ----------
        json_file : str, optional
            _description_, by default None

        Raises
        ------
        ValueError
            _description_
        """
        if json_file is None:
            json_file = self.json_file

        defaults = {k: v.default for i in self.__class__.__mro__[:-1][::-1]
                    for k, v in
                    inspect.signature(i.__init__).parameters.items()}
        output = {k: v for k, v in self.__dict__.items()
                  if v is not defaults[k]}
        output = {k: (list(v) if isinstance(v, np.ndarray) else v)
                  for k, v in output.items()}
        if 'name' not in output:
            raise ValueError('Must have a name to be saved to file')
        with open(json_file, 'r+', encoding='UTF-8') as f:
            try:
                file_data = json.load(f)
                if self.__class__.__name__ not in file_data:
                    file_data[self.__class__.__name__] = []
                # correct_objects = file_data[self.__class__.__name__]
                if output['name'] in [i['name'] for i in file_data[self.__class__.__name__]]:
                    raise ValueError(
                        f'A {self.__class__.__name__} with the name {self.name} is already in the json file, choose a different name')
                file_data[self.__class__.__name__].append(output)
                f.seek(0)
                json.dump(file_data, f, indent=4)
            except json.JSONDecodeError:
                json.dump({self.__class__.__name__:[output]}, f, indent=4)

    # TODO will defo have to change this when with inheritance i reckon accs mayb not
    def _load(self, json_file: str = None) -> None:
        """_summary_

        Parameters
        ----------
        json_file : str, optional
            _description_, by default None
        """
        if json_file is None:
            json_file = self.json_file
        with open(json_file, 'r', encoding='UTF-8') as f:
            try:
                file_data = json.loads(f.read())
                correct_objects = file_data[self.__class__.__name__]
                correct_object = next(
                    (i for i in correct_objects if i['name'] == self.name), None)
                try:
                    for key, value in correct_object.items():
                        if isinstance(value, list):
                            value = np.array(value)
                        setattr(self, key, value)
                except AttributeError:
                    warnings.warn(
                        f'name {self.name} not found in json file {json_file}')
            except json.JSONDecodeError:
                pass


# # [
# #     {
# #         "name": "joe"
# #     },
# #     {
# #         "name": "idk"
# #     },
#     {
#         "name": "flatPanelSimap",
#         "dimensions": [
#             300.0,
#             300.0
#         ],
#         "pixel_size": [
#             50.0,
#             50.0
#         ],
#         "psf": 1.0,
#         "bin_thresholds": [
#             20.0,
#             30.0,
#             39.75
#         ],
#         "scintillator_material": "CsI",
#         "scintillator_thickness": 600.0
#     },
# #     {
# #         "name": "heloo"
# #     }
# # ]