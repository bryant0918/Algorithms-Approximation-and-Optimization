from binary_trees import *
#test file for binary search trees remove function.


def test_leaf():
    ''' this is a test function for if the target to remove is the root of the tree
    '''
    print("testing leaf")
    #test correct removal for leaf node that is also root
    bst = BST()
    bst.insert(1)
    bst.remove(1)
    removed = bst.__str__()
    assert removed == '[]'

    #test correct removal for leaf node that is to the left of its parent

    bst.insert(10)
    bst.insert(8)
    bst.remove(8)

    removed = bst.__str__()
    assert removed == '[10]'

    #test correct removal for leaf node that is to the right of its parent
    bst.insert(12)
    bst.remove(12)

    removed = bst.__str__()
    assert removed == '[10]'


def test_two_children():
    '''This function tests the correct removal procedure for a target that has two children
    '''
    bst = BST()
    nodes = [4, 2, 1, 3, 10, 5, 6, 9, 7, 11, 15, 14, 16, 12]

    for node in nodes:
        bst.insert(node)

    #test removal of root
    bst.remove(4)
    removed = bst.__str__()
    assert removed == '[3]\n[2, 10]\n[1, 5, 11]\n[6, 15]\n[9, 14, 16]\n[7, 12]'

    #test removal of non root
    bst.remove(10)
    removed = bst.__str__()
    assert removed == '[3]\n[2, 9]\n[1, 5, 11]\n[6, 15]\n[7, 14, 16]\n[12]'

    #test removal of root again
    bst.remove(3)
    removed = bst.__str__()
    assert removed == '[2]\n[1, 9]\n[5, 11]\n[6, 15]\n[7, 14, 16]\n[12]'

    #test removal of another non root
    bst.remove(15)
    removed = bst.__str__()
    assert removed == '[2]\n[1, 9]\n[5, 11]\n[6, 14]\n[7, 12, 16]'


def test_one_child():
    raise NotImplementedError

if __name__ == "__main__":
    test_leaf()
    test_two_children()
    pass