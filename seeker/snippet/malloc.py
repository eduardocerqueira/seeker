#date: 2025-04-03T16:53:32Z
#url: https://api.github.com/gists/499e2dc3220fbe2e013abc363ef8cd7f
#owner: https://api.github.com/users/karannparikh

from sortedcontainers import SortedList
import bisect
import copy

class SpaceBlock:
    def __init__(self, offset, size):
        self.offset = offset
        self.size = size
        self.head = offset
        self.tail = offset + size

    def __lt__(self, other):
        return self.size < other.size

    def __repr__(self):
        return f"SpaceBlock(offset={self.offset}, size={self.size} head={self.head} tail={self.tail})"

class MemoryManager:
    def __init__(self, N):
        self.N = N
        self.space_map = SortedList()
        self.head_map = {}
        self.tail_map = {}
        self.a_map = {}
        self.__add_new_block__(SpaceBlock(offset=0, size=N))

    def __add_new_block__(self, block):
        if block.offset + block.size < block.tail:
            breakpoint()
        self.space_map.add(block)
        self.head_map[block.head] = block
        self.tail_map[block.tail] = block

    def __remove_block__(self, block):
        print(f"Removing block: {block}")
        self.space_map.remove(block)
        del self.head_map[block.head]
        del self.tail_map[block.tail]

    def malloc(self, size):
        index = bisect.bisect_left(self.space_map, SpaceBlock(offset=0, size=size))
        if index >= len(self.space_map):
            return None
        block = self.space_map[index]
        self.__remove_block__(block)
        if block.size > size:
            self.__add_new_block__(SpaceBlock(offset=block.offset + size, size=block.size - size))
        block = SpaceBlock(offset=block.offset, size=size)
        self.a_map[block.offset] = block
        return block.offset

    def debug(self):
        print("Space Map:")
        for block in self.space_map:
            print(block.offset, block.size)
        print("Head Map:")
        print(self.head_map)
        print("Tail Map:")
        print(self.tail_map)
        print("A Map:")
        print(self.a_map)


    def free(self, offset):
        if offset not in self.a_map:
            raise Exception("Invalid offset")
        new_block = self.a_map[offset]
        print(f"Freeing a block at offset: {new_block}")
        del self.a_map[offset]

        if new_block.head in self.tail_map:
            head_block = self.tail_map[new_block.head]
            self.__remove_block__(head_block)
            new_block = SpaceBlock(offset=head_block.offset, size=head_block.size + new_block.size)

        if new_block.tail in self.head_map:
            tail_block = self.head_map[new_block.tail]
            del self.head_map[new_block.tail]
            self.__remove_block__(tail_block)
            new_block = SpaceBlock(offset=new_block.offset, size=new_block.size + tail_block.size)
        
        self.__add_new_block__(new_block)

def test_memory_manager():
    # Initialize memory manager with 100 units
    mm = MemoryManager(100)
    
    # Test basic malloc
    ptr1 = mm.malloc(20)
    assert ptr1 == 0, "First allocation should start at 0"
    assert len(mm.space_map) == 1, "Should have one free block remaining"
    
    # Test second malloc
    ptr2 = mm.malloc(30)
    assert ptr2 == 20, "Second allocation should start after first block"
    
    # Test malloc when space is available but fragmented
    ptr3 = mm.malloc(40)
    assert ptr3 == 50, "Third allocation should start after second block"
    
    # Test malloc when not enough space
    ptr4 = mm.malloc(20)
    assert ptr4 is None, "Should return None when no space available"
    
    # Test basic free
    mm.free(ptr1)  # Free first block
    assert len(mm.space_map) == 2, "Should merge with adjacent free space"
    
    # Test free and merge both sides
    ptr5 = mm.malloc(10)  # Allocate in the freed space
    mm.free(ptr2)  # Free middle block
    mm.free(ptr5)  # This should merge with both adjacent blocks
    assert len(mm.space_map) == 2, "Should have two free blocks after merges"
    try:
        mm.free(ptr5)
        assert False, "Should raise an exception"
    except Exception as e:
        print("Expected Exception raised: ", e)

if __name__ == "__main__":
    test_memory_manager()



