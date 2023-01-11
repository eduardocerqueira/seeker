#date: 2023-01-11T17:16:09Z
#url: https://api.github.com/gists/be9fc8f2d99801a2d20004ba904e29cc
#owner: https://api.github.com/users/gabonator

# Raw string literals is evil! All types of line endings (\x0d\x0a, \x0d, \x0a) are 
# converted into native line ending character (on linux "\x0a"). Contrary to this fact
# the documentation https://en.cppreference.com/w/cpp/language/string_literal says:
#
#   ...Used to avoid escaping of any character. Anything between the delimiters 
#   becomes part of the string.
#
# Here is a proof, lets create 3 strings using various line ending characters/sequences
# inside raw string literal constant. Print out the length and contents for each
# The first is using \r\n sequence, second \x0d, third \x0a. So we would expect
# that the first length will be larger than the others, but surprisingly all of them
# are the same! Hexadecimal dump reveals that the line endings were converted into \x0a
# for all three cases!
#
#
# Output:
#
#  lengths: 23, 23, 23, 23
#  6c 69 6e 65 31 0a 6c 69 6e 65 32 0a 6c 69 6e 65 33 0a 6c 69 6e 65 34
#  6c 69 6e 65 31 0a 6c 69 6e 65 32 0a 6c 69 6e 65 33 0a 6c 69 6e 65 34
#  6c 69 6e 65 31 0a 6c 69 6e 65 32 0a 6c 69 6e 65 33 0a 6c 69 6e 65 34
#  6c 69 6e 65 31 40 6c 69 6e 65 32 40 6c 69 6e 65 33 40 6c 69 6e 65 34
#
#  00000000: 72 2a 20 6d 73 67 31 20 3d 20 52 22 28 6c 69 6e  r* msg1 = R"(lin
#  00000010: 65 31 0d 0a 6c 69 6e 65 32 0d 0a 6c 69 6e 65 33  e1..line2..line3
#  00000020: 0d 0a 6c 69 6e 65 34 29 22 3b 0a 63 6f 6e 73 74  ..line4)";.const
#  00000030: 20 63 68 61 72 2a 20 6d 73 67 32 20 3d 20 52 22   char* msg2 = R"
#  00000040: 28 6c 69 6e 65 31 0d 6c 69 6e 65 32 0d 6c 69 6e  (line1.line2.lin
#  00000050: 65 33 0d 6c 69 6e 65 34 29 22 3b 0a 63 6f 6e 73  e3.line4)";.cons
#  00000060: 74 20 63 68 61 72 2a 20 6d 73 67 33 20 3d 20 52  t char* msg3 = R
#  00000070: 22 28 6c 69 6e 65 31 0a 6c 69 6e 65 32 0a 6c 69  "(line1.line2.li
#  00000080: 6e 65 33 0a 6c 69 6e 65 34 29 22 3b 0a 63 6f 6e  ne3.line4)";.con
#  00000090: 73 74 20 63 68 61 72 2a 20 6d 73 67 34 20 3d 20  st char* msg4 =
#  000000a0: 52 22 28 6c 69 6e 65 31 40 6c 69 6e 65 32 40 6c  R"(line1@line2@l
#  000000b0: 69 6e 65 33 40 6c 69 6e 65 34 29 22 3b 0a 76 6f  ine3@line4)";.vo

set -e
cat > test1.cpp <<- EOM
#include <stdio.h>
#include <string.h>
const char* msg1 = R"(line1!line2!line3!line4)";
const char* msg2 = R"(line1|line2|line3|line4)";
const char* msg3 = R"(line1^line2^line3^line4)";
const char* msg4 = R"(line1@line2@line3@line4)";
void myprint(const char* msg)
{
  while (*msg)
    printf("%02x ", *msg++);
  printf("\n");
}
int main(void)
{
  printf("lengths: %ld, %ld, %ld, %ld\n", strlen(msg1), strlen(msg2), strlen(msg3), strlen(msg4));
  myprint(msg1);
  myprint(msg2);
  myprint(msg3);
  myprint(msg4);

  return 0;
}
EOM

cat test1.cpp | sed 's_!_\x0d\x0a_g' \
 | sed 's_\^_\x0a_g' \
 | sed 's_|_\x0d_g' > test.cpp

gcc test.cpp -o test
./test

dd if=test.cpp of=test2.cpp bs=1 skip=$((0x30)) count=$((0xc0)) 2> /dev/null
xxd -g 1 test2.cpp
rm test test.cpp test1.cpp test2.cpp
