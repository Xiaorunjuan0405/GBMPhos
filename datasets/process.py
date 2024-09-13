
def replace_numbers(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        count = 1
        for line in infile:
            if line.startswith('>'):
                line = f'>{count}\n'
                count += 1
            outfile.write(line)

# 使用示例
replace_numbers('ST-test.FASTA', 'output.fasta')