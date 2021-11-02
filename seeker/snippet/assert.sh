#date: 2021-11-02T17:03:52Z
#url: https://api.github.com/gists/ce89cd38ca71d496c43851eda58523ae
#owner: https://api.github.com/users/lebidouilleur

### assert.sh ---                                               -*- mode: sh; -*-

## Copyright (C) 2021  lebidouilleur

## Author: lebidouilleur <lebidouilleur@icloud.com>
## Keywords:

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

### Commentary:

## ⚠ This simple 'library' intentionally does not support 'advanced' features. ⚠

### Code:


function __assert_expr {
  if ! test "$@"
  then
    return $_EXIT_FAIL
  else
    return $_EXIT_SUCCESS
  fi
}


function assert_eq {
  __assert_expr $1 == $2
}


function assert_neq {
  __assert_expr $1 != $2
}


function assert_gt {
  __assert_expr $1 > $2
}


function assert_geq {
  __assert_expr $1 >= $2
}


function assert_lt {
  __assert_expr $1 < $2
}


function assert_leq {
  __assert_expr $1 <= $2
}


function assert_true {
  assert_eq true $1
}


function assert_false {
  assert_eq false $1
}


function assert_directory_exists {
  __assert_expr -d "$1"
}


function assert_file_exists {
  __assert_expr -f "$1"
}


function assert_executable {
  __assert_expr -x "$1"
}


function assert_readable {
  __assert_expr -r "$1"
}


function assert_writable {
  __assert_expr -w "$1"
}


# Usage:
# declare -a firstArray  = ('a' 'b' 'c')
# declare -a secondArray = ('a' 'b' 'c') 
# assert_array_eq firstArray[@] secondArray[@]
function assert_array_eq {
  declare -a first=(${!1})
  declare -a second=(${!2})

  if assert_neq ${#first[@]} ${#second[@]}
  then
    return $_EXIT_FAIL
  fi

  for iterator in $(seq 0 $((${#first[@]}-1)))
  do
    if assert_neq ${first[$iterator]} ${second[$iterator]}
    then
      return $_EXIT_FAIL
    fi
  done
}


# Usage:
# declare -a firstArray  = ('a' 'b' 'c')
# declare -a secondArray = ('a' 'b' 'c') 
# assert_array_neq firstArray[@] secondArray[@]
function assert_array_neq {
  ! assert_array_eq $1[@] $2[@]
}


function assert_string_contains {
  assert_neq "${1#*$2}" "$1"
}


_EXIT_SUCCESS=0
_EXIT_FAIL=1
