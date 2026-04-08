import warnings
import numpy as np

class DataArray:
    """
    The point of this class is to preinitialise a data array with a shape based on the fixed sizes and assumed maximum variable dimension sizes to avoid pythons dynamic array resizing which can be very slow. """
    def __init__(self,
                 fixed_sizes: list[int], # determines shape of the input vectors
                 assumed_maximum_variable_dimension_sizes: list[int], # dimensions the input vectors are added to
                 input_data_array: np.ndarray = np.array([]),
                 dimension_order: list[str] = []
                 ) -> None:
        
        self.fixed_sizes = fixed_sizes
        self.assumed_maximum_variable_dimension_sizes = assumed_maximum_variable_dimension_sizes
        self.input_data_array = input_data_array
        self.total_dimensions = len(self.fixed_sizes) + len(self.assumed_maximum_variable_dimension_sizes)
        self.dimension_order = dimension_order
        
        if len(dimension_order) != self.total_dimensions:
            warnings.warn(f"Dimension order length {len(dimension_order)} does not match total dimensions {self.total_dimensions}. Defaulting to fixed size dimensions then variable dimensions in order of input.")
            self.dimension_order = [f'fixed_{i}' for i in range(len(self.fixed_sizes))] + [f'variable_{i}' for i in range(len(self.assumed_maximum_variable_dimension_sizes))]

        self.create_zeros_data_array()
        self.data_array[*(slice(i) for i in self.input_data_array.shape)] = self.input_data_array
    
    def create_zeros_data_array(self) -> None:
        """Create a data array with the appropriate shape based on the fixed sizes, total dimensions, and assumed maximum variable dimension size. i.e. allocating the maximum amount of memory that we might need to store the dataset vectors and distance matrix to avoid dynamic resizing which can be very slow."""
        
        if hasattr(self, 'data_array') and self.data_array is not None and self.data_array.size != 0:
            raise ValueError("Data array already exists. Cannot create a new one.")
        
        input_tuple= ()
        for dimension in self.dimension_order:
            if 'fixed' in dimension:
                size = self.fixed_sizes[int(dimension.split('_')[1])]
            elif 'variable' in dimension:
                size = self.assumed_maximum_variable_dimension_sizes[int(dimension.split('_')[1])]
            else:
                raise ValueError(f"Invalid dimension name {dimension} in dimension order. Must contain 'fixed' or 'variable'.")
            input_tuple += (size,)
        
        self.data_array = np.zeros(input_tuple)
    
    def _ensure_input_allowed(self, input_vector: np.ndarray, input_indices: tuple, allow_index_overflow: bool = True) -> None:
        """Ensure that the input vector has the correct shape and dimensions."""
            
        fixed_dimensions_ordered_shape_dict = {d: self.fixed_sizes[self.dimension_order.index(d)] for d in self.dimension_order if 'fixed' in d}
        print(f"Fixed dimensions ordered shape dict: {fixed_dimensions_ordered_shape_dict}")
        for i in range(len(fixed_dimensions_ordered_shape_dict)):
            if input_vector.shape[i] != list(fixed_dimensions_ordered_shape_dict.values())[i]:
                raise ValueError(f"Input vector shape {input_vector.shape} does not match expected fixed sizes {list(fixed_dimensions_ordered_shape_dict.values())}.")
                    
        _index_overflow = False
        input_indices_lengths = [input_indices[i].stop - input_indices[i].start for i in range(len(input_indices))]
        if np.any(np.array(input_indices_lengths) >= np.array(self.assumed_maximum_variable_dimension_sizes)):
            warnings.warn(f"Input index {i} exceeds the maximum size for dimension {self.dimension_order[i]} in the data array. "
                            f"If index overflow is allowed, the size of the data array along this dimension will be doubled to accommodate new input vectors.\n"
                            f"Consider increasing the assumed maximum variable dimension sizes to avoid this warning in the future.")     
            self.assumed_maximum_variable_dimension_sizes[input_indices.index(i)] *= 2
            _index_overflow = True
        
        if _index_overflow:
            
            if allow_index_overflow:
                old_array = self.data_array.copy()
                self.data_array = np.array([])
                self.create_zeros_data_array()
                self.data_array[*(slice(i) for i in old_array.shape)] = old_array
            else:
                raise ValueError("Input indices exceed the maximum size for the data array and allow_index_overflow is set to False.")
                
        if np.any(self.data_array[input_indices] != 0):
            raise ValueError("Input vector cannot be added to the data array because the corresponding indices are already occupied.")
    
    def add_input_vector_to_data_array(self, input_vector: np.ndarray, input_indices: tuple, allow_index_overflow: bool = True) -> None:
        """Add the input vector to the data array at the specified indices."""
        self._ensure_input_allowed(input_vector, input_indices, allow_index_overflow)
        self.data_array[input_indices] = input_vector

    def remove_unoccupied_vectors(self, max_dimension_tuple: tuple) -> np.ndarray:
        """Remove any unoccupied vectors from the data array."""
        if self.data_array is None or self.data_array.size == 0:
            raise ValueError("Data array has not been initialized.")
        
        mask = tuple(slice(0, dim) for dim in max_dimension_tuple)
        return self.data_array[mask]