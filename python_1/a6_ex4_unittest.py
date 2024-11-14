from a6_ex4 import permute

test_inputs = ['', 'a', 'ab', 'abc']
expected_outputs = [{''},
                    {'a'},
                    {'ab', 'ba'},
                    {'cba', 'abc', 'bac', 'acb', 'cab', 'bca'}]

for index, (test_input, expected_output) in enumerate(zip(test_inputs, expected_outputs), 1):
    output = permute(test_input)
    assert output == expected_output, f"Test {index} failed. input={test_input}, output={output}, expected_output={expected_output}"

print("OK")