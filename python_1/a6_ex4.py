def permute(s: str) -> set[str]:
    # Base case: if the string is empty, return a set with an empty string
    if len(s) == 0:
        return {''}

    # Recursive case: find all permutations of the string
    result = set()
    for i in range(len(s)):
        # Fix the character at index i and permute the rest of the string
        remaining = s[:i] + s[i + 1:]  # String without the character at index i
        for perm in permute(remaining):  # Recursively permute the remaining string
            result.add(s[i] + perm)  # Prepend the current character to each permutation

    return result
