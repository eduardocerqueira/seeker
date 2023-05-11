#date: 2023-05-11T16:44:29Z
#url: https://api.github.com/gists/f65638e1a7296dd97925cc5d4621c655
#owner: https://api.github.com/users/mypy-play

from typing import Literal, Optional, TypeAlias
from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class X:
    pass
# Want to explain why eq and frozen would be set
@dataclass(eq=True, frozen=True)
class O:
    pass

Token : "**********"

Cell : "**********"

GridPos : TypeAlias = Literal[1,2,3,4,5,6,7,8,9]

# GridPos -> Cell
Grid : TypeAlias = dict[GridPos, Cell]

# initGrid : Void -> Grid
def initGrid() -> Grid:
    grid: Grid = {
        1:None, 2:None, 3:None,
        4:None, 5:None, 6:None,
        7:None, 8:None, 9:None
        
    }
    return grid

# getCell : Grix x GridPos -> Cell option
def getCell(grid: Grid, pos: GridPos) -> Optional[Cell]:
    if pos not in grid:
        return None
        
    return grid[pos]
    
# placeToken : "**********"
def placeToken(token: "**********": GridPos, grid: Grid) -> Optional[Grid]:
    if getCell(grid, pos) != None:
        return None
    
    new_grid = grid.copy()
    new_grid[pos] = "**********"
    return new_grid

# gridToString : Grid -> String
def gridTostring(grid: Grid) -> str:
    return f"""
    {cellToString(getCell(grid, 1))}|{cellToString(getCell(grid, 2))}|{cellToString(getCell(grid, 3))}
    {cellToString(getCell(grid, 4))}|{cellToString(getCell(grid, 5))}|{cellToString(getCell(grid, 6))}
    {cellToString(getCell(grid, 7))}|{cellToString(getCell(grid, 8))}|{cellToString(getCell(grid, 9))}
    """

# cellToString : Cell -> String
def cellToString(cell: Cell) -> str:
    if isinstance(cell, (X, O)):
        return tokenToString(cell)
    return " "

# tokenToString : "**********"
def tokenToString(token: "**********":
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"i "**********"s "**********"i "**********"n "**********"s "**********"t "**********"a "**********"n "**********"c "**********"e "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"X "**********") "**********": "**********"
        return "X"
    return "O"
