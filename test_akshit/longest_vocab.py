def find_longest_token(file_path):
    # Initialize variables to store the longest token and its length
    longest_token = ''
    max_length = 0
    import pdb; pdb.set_trace()
    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line to get the token part (after the space)
            parts = line.split()
            if len(parts) > 1:  # To ensure there's a token part
                token = parts[1]  # The token is the second part
                # Update the longest token if the current one is longer
                if len(token) > max_length:
                    longest_token = token
                    max_length = len(token)
    
    return longest_token

# Path to your vocabulary file
file_path = 'TinyStoriesV2-GPT4-train.txt_my_vocab.json'

# Find and print the longest token
longest = find_longest_token(file_path)
print("The longest token is:", longest)
