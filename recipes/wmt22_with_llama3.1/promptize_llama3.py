# Input and output file paths
input_file_path = "newstest2022-src.en"
output_file_path = "newstest2022-src-prompt.en"

# Open the input file for reading and the output file for writing
with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file:
    # Loop through each line in the input file
    for line in input_file:
        # Set up the prompt provided by Lama3 with the current line's content
        # PLEASE note the space between "<|start_header_id|> user"
        # and "<|start_header_id|> assistant"
        # This is due to a specific behavior of onmt_tokenize
        # HF tokenizer has a similar strange behavior:
        # https://github.com/huggingface/transformers/issues/31513#issuecomment-2340151976
        prompt = f"<|start_header_id|> user<|end_header_id|>｟newline｠Translate the following text from English into German.｟newline｠English: {line.strip()}｟newline｠German:<|eot_id|><|start_header_id|> assistant<|end_header_id|>｟newline｠"  # noqa: E501
        output_file.write(prompt + "\n")
