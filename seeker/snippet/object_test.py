#date: 2024-09-23T17:00:26Z
#url: https://api.github.com/gists/669a2377ef929c935f0236210e8b6960
#owner: https://api.github.com/users/wmcdonald404

#!/usr/bin/env python3
#
# pytest --tb=line -v object_test.py --setup-show

import pytest
import os

class Fruit:
    def __init__(self,colour,shape):
        self.colour=colour
        self.shape=shape

    def create(self):
        if not os.path.exists('/tmp/' + self.colour):
            os.mknod('/tmp/' + self.colour)

    def remove(self):
        if os.path.exists('/tmp/' + self.colour):
            os.remove('/tmp/' + self.colour)

@pytest.fixture(scope="module")
def create_object_instance():
    orange = Fruit('Orange', 'Round')
    orange.create() #creates state outside the object
    return orange

def test_created_object_type(create_object_instance):
    test_object=create_object_instance
    assert type(test_object) is Fruit

def test_created_object_colour(create_object_instance):
    test_object=create_object_instance
    assert test_object.colour == 'Orange'

def test_created_object_shape(create_object_instance):
    test_object=create_object_instance
    assert test_object.shape == 'Round'

def test_created_object_file(create_object_instance):
    test_object=create_object_instance
    assert os.path.exists('/tmp/' + test_object.colour)

@pytest.fixture()
def remove_object_instance():
    orange = Fruit('Orange', 'Round')
    orange.remove() #remove state outside the object
    return orange

def test_removed_object_type(remove_object_instance):
    test_object=remove_object_instance
    assert not os.path.exists('/tmp/' + test_object.colour)
