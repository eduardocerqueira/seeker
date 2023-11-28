//date: 2023-11-28T16:50:21Z
//url: https://api.github.com/gists/6e7de0e3e9664fa1bfbe3ed2f6db9283
//owner: https://api.github.com/users/gomyway1216

class Solution {
    private final Map<Character, String> mapping = Map.of(
        '2', "abc", '3', "def",
        '4', "ghi", '5', "jkl",
        '6', "mno", '7', "pqrs",
        '8', "tuv", '9', "wxyz"
    );

    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if (digits == null || digits.length() == 0) {
            return combinations;
        }

        backtrack(combinations, new StringBuilder(), digits, 0);
        return combinations;
    }

    private void backtrack(List<String> combinations, StringBuilder current, String digits, int index) {
        if (index == digits.length()) {
            combinations.add(current.toString());
            return;
        }

        String chars = mapping.get(digits.charAt(index));
        for (char letter : chars.toCharArray()) {
            current.append(letter);
            backtrack(combinations, current, digits, index + 1);
            current.deleteCharAt(current.length() - 1);
        }
    }
}