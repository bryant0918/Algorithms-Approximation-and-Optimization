# test_specs.py
"""Python Essentials: Unit Testing.
Bryant McArthur
Math 321
September 23
"""

import specs
import pytest



def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    assert specs.smallest_factor(2) == 2, "failed for n = 2"
    assert specs.smallest_factor(9) == 3, "failed for n = 9"
    
    
#test_smallest_factor()
    

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    assert specs.month_length("January") == 31, "Failed for January"
    assert specs.month_length("January", True) == 31, "Failed for January leap year"
    assert specs.month_length("February") == 28, "Failed for Feb"
    assert specs.month_length("February", True) == 29, "Failed for Feb leap year"
    assert specs.month_length("September") == 30, "Failed for September"
    assert specs.month_length("january") == None, "Failed for valueerror"
    
#test_month_length()
    

# Problem 3: write a unit test for specs.operate().
def test_operate():
    with pytest.raises(TypeError) as excinfo:
        specs.operate(1,2,3)
    assert excinfo.value.args[0] == "oper must be a string"
    assert specs.operate(1,2,"+") == 3, "Add failed"
    assert specs.operate(1,2,"-") == -1, "Subtract failed"
    assert specs.operate(1,2,"*") == 2, "Multiply failed"
    assert specs.operate(1,2,"/") == 0.5, "Divide failed"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(2, 0, "/")
    assert excinfo.value.args[0] == "division by zero is undefined"
    
    
#test_operate()

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction(5,0)
    assert excinfo.value.args[0] == "denominator cannot be zero"
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction("s",4)
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    frac = specs.Fraction(5,1)
    assert str(frac) == "5"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.
    frac = specs.Fraction(5,1) 
    assert float(frac) == 5.0
    

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_1_2 == float(frac_1_2)
    

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 + frac_1_3 == 5/6
    assert frac_1_3 + frac_n2_3 == -1/3
    

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 - frac_1_3 == 1/6
    assert frac_n2_3 - frac_1_3 == -1


def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 * frac_1_3 == 1/6
    assert frac_n2_3 * frac_1_3 == -2/9


def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 / frac_1_3 == 3/2
    frac1 = specs.Fraction(0,4)
    frac2 = specs.Fraction(4,0)
    """
    with pytest.raises(ZeroDivisionError) as excinfo:
        frac_1_2 / frac2
    assert excinfo.value.args[0] == "cannot divide by zero"
    """














# Problem 5: Write test cases for Set.
