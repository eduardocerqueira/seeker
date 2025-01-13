#date: 2025-01-13T16:53:15Z
#url: https://api.github.com/gists/eaa83883850964783c9911747b69f0f8
#owner: https://api.github.com/users/alxkahovsky

import unittest


class TreeStore:
    def __init__(self, items):
        self.items = {item['id']: item for item in items}
        self.children = {}

        for item in items:
            parent_id = item.get('parent')
            if parent_id not in self.children:
                self.children[parent_id] = []
            self.children[parent_id].append(item)

    def getAll(self):
        return list(self.items.values())

    def getItem(self, id):
        return self.items.get(id)

    def getChildren(self, id):
        return self.children.get(id, [])

    def getAllParents(self, id):
        parents = []
        current_item = self.items.get(id)

        while current_item:
            parents.append(current_item)
            parent_id = current_item.get('parent')
            current_item = self.items.get(parent_id)

        return parents[1:]


items_list = [
    {"id": 1, "parent": "root"},
    {"id": 2, "parent": 1, "type": "test"},
    {"id": 3, "parent": 1, "type": "test"},
    {"id": 4, "parent": 2, "type": "test"},
    {"id": 5, "parent": 2, "type": "test"},
    {"id": 6, "parent": 2, "type": "test"},
    {"id": 7, "parent": 4, "type": None},
    {"id": 8, "parent": 4, "type": None}
]

ts = TreeStore(items_list)


print(ts.getAll())
print(ts.getItem(7))
print(ts.getChildren(4))
print(ts.getChildren(5))
print(ts.getAllParents(7))


class TestTreeStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items = items_list
        cls.ts = TreeStore(cls.items)

    def test_getAll(self):
        self.assertEqual(self.ts.getAll(), self.items)

    def test_getItem(self):
        self.assertEqual(self.ts.getItem(7), {"id": 7, "parent": 4, "type": None})
        self.assertEqual(self.ts.getItem(1), {"id": 1, "parent": "root"})
        self.assertIsNone(self.ts.getItem(999))

    def test_getChildren(self):
        self.assertEqual(self.ts.getChildren(4), [
            {"id": 7, "parent": 4, "type": None},
            {"id": 8, "parent": 4, "type": None}
        ])
        self.assertEqual(self.ts.getChildren(2), [
            {"id": 4, "parent": 2, "type": "test"},
            {"id": 5, "parent": 2, "type": "test"},
            {"id": 6, "parent": 2, "type": "test"}
        ])
        self.assertEqual(self.ts.getChildren(5), [])
        self.assertEqual(self.ts.getChildren(999), [])

    def test_getAllParents(self):
        self.assertEqual(self.ts.getAllParents(7), [
            {"id": 4, "parent": 2, "type": "test"},
            {"id": 2, "parent": 1, "type": "test"},
            {"id": 1, "parent": "root"}
        ])
        self.assertEqual(self.ts.getAllParents(2), [
            {"id": 1, "parent": "root"}
        ])
        self.assertEqual(self.ts.getAllParents(1), [])
        self.assertEqual(self.ts.getAllParents(999), [])

if __name__ == "__main__":
    unittest.main()