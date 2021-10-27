import sys
import re

def create_vocab_file(file_path, output_file):
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    added_strings = set()
    with open(output_file, 'w') as outfile:
        for factor in factors:
            subbed = re.sub("[a-z](?!\(*[a-z]+)", "x", factor)
            if subbed not in added_strings:
                outfile.write(subbed + "\n")
                added_strings.add(subbed)
        for expansion in expansions:
            subbed = re.sub("[a-z](?!\(*[a-z]+)", "x", expansion)
            if subbed not in added_strings:
                outfile.write(subbed + "\n")
                added_strings.add(subbed)
    outfile.close()

def main():
    create_vocab_file("train.txt", "vocab.txt")

if __name__ == "__main__":
    main()
