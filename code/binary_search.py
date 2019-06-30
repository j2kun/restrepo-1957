from dataclasses import dataclass
from typing import Optional


@dataclass
class BinarySearchHint:
    '''Class representing the inremental step of a binary search test.'''

    '''True if the target of the search has been found.'''
    found: bool = False

    '''True if the parameter value was determined to be too low.'''
    tooLow: bool = False


@dataclass
class BinarySearchResult:
    '''Class representing the output of a binary search.'''

    '''True if the target of the search has been found.'''
    found: bool = True

    '''If found=True, this field contains the output value of the search.

    Otherwise, the value is undefined, but set to None by default. If
    found=True and value=None, then None is the value found during the search.
    '''
    value: Optional[float] = None


def binary_search(test, param_min=0, param_max=1, tolerance=1e-9, callback=None):
    '''
    Perform a binary search for a given parameter value.

    Call the type of the value being searched for T.

    Args:
    - test: a callable T -> BinarySearchHint
    - param_min: the smallest legal value of the parameter being searched for
    - param_max: the largest legal value of the parameter being searched for
    - tolerance: the numerical tolerance for termination
    - callback: an arbitrary callback executed at the start of each search loop

    Returns:
      An instance of BinarySearchResult
    '''
    current_min = param_min
    current_max = param_max

    while current_max - current_min > tolerance:
        tested_value = (current_max + current_min) / 2
        if callback:
            callback(dict(
                current_min=current_min,
                current_max=current_max,
                tested_value=tested_value
            ))
        hint = test(tested_value)
        if hint.found:
            return BinarySearchResult(found=True, value=tested_value)
        elif hint.tooLow:
            current_min = tested_value
        else:
            current_max = tested_value

    return BinarySearchResult(found=False, value=None)
