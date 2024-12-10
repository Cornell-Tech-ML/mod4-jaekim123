class Tensor:
    # ... existing methods ...
    
    def __abs__(self) -> 'Tensor':
        return self.apply_elementwise(abs)
    
    def dim(self) -> int:
        return len(self.shape)
    
    def all(self) -> 'Tensor':
        # Implement method to return Tensor with a single boolean indicating if all elements are True
        return self.reduce(lambda a, b: a and b, initial=True)
    
    def item(self) -> bool:
        # Implement method to extract the single boolean value from the Tensor
        return self.storage[0] == 1 