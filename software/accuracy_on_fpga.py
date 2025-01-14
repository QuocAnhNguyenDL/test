# Function to calculate the accuracy based on the given rules
def accuracy(dataset, result):
    try:
        # Read the contents of both files
        with open(dataset, 'r') as dataset, open(result, 'r') as result:
            lines_a = dataset.readlines()
            lines_b = result.readlines()

        # Check if the files have the same number of lines
        if len(lines_a) != len(lines_b):
            raise ValueError("Files do not have the same number of lines.")

        # Count the number of correct lines
        correct_count = 0
        total_lines = len(lines_a)

        for line_a, line_b in zip(lines_a, lines_b):
            line_a = line_a.strip()
            line_b = line_b.strip()

            if (line_a.startswith("/home/quocna/project/DOAN2/dataset/dataset/test/no-Fire") and line_b == '1') or (line_a.startswith("/home/quocna/project/DOAN2/dataset/dataset/test/Fire") and line_b == '0'):
                correct_count += 1

        # Calculate accuracy as a percentage
        accuracy = (correct_count / total_lines) * 100 if total_lines > 0 else 0

        return accuracy

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
file_a_path = "/home/quocna/project/DOAN2/fire_detec/software/txt/dataset.txt"
file_b_path = "/home/quocna/project/DOAN2/fire_detec/software/txt/result_fpga.txt"
accuracy = accuracy(file_a_path, file_b_path)

if accuracy is not None:
    print(f"Accuracy for model on FPGA: {accuracy:.2f}%")