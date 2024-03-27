import ast
from Preprocessing_400 import number_of_data
file_path_reflection = f"Refraction_values_{int(number_of_data/1000)}k.txt"
file_path_thickness = f"Thickness_values_{int(number_of_data/1000)}k.txt"
file_path_materials = f"Materials_list_{int(number_of_data/1000)}k.txt"


def process_reflection(file_path):
    data_list = []
    current_data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading and trailing whitespaces including '\n'

            # Split the line based on '[' and ' ]'
            split_line = [s.strip() for s in line.split('[,]')]
            for part in split_line:
                # Process each part to extract float numbers separated by space
                for num in part.split():
                    num = num.strip('][')  # Remove '[' and ']' characters
                    if num:
                        try:
                            float_number = float(num)
                            current_data.append(float_number)
                        except ValueError:
                            pass

            if ']' in line:
                # End of data section, append current_data to data_list
                data_list.append(current_data)
                current_data = []  # Reset current_data for the next section

    return data_list


def process_thickness(file_path):
    """
    Read lists of numbers from a text file.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list of lists: A list containing the lists of numbers read from the file.
    """
    lists = []

    with open(file_path, 'r') as file:
        for line in file:
            # Remove square brackets and split the line into individual numbers
            numbers = line.strip()[1:-1].split()
            # Convert each number from string to float and add to the list
            numbers = [float(num) for num in numbers]
            lists.append(numbers)

    return lists


def process_materials_pre(file_path):
    """
    Read lists of lists from a file and return them as a list of lists.

    Args:
    file_path (str): The path to the file containing the lists of lists.

    Returns:
    list: A list of lists read from the file.
    """
    # Open the file
    with open(file_path, 'r') as file:
        # Read the contents of the file
        contents = file.readlines()

    # Strip newline characters from each line and return as list of strings
    return [line.strip() for line in contents]


def process_material(file_path):
    result_mat = process_materials_pre(file_path)
    result_mat_modified = [", ".join(mat.replace(".", "").split()) for mat in result_mat]

    # Print out result_mat_modified for debugging

    res_mat__ = [s for s in result_mat_modified]
    rm = "".join(res_mat__)
    rm_modified = rm.replace("]][[", "]],[[")
    rm_modified = rm_modified.replace("][", "],[")
    # Convert the string representation into nested list
    result_material = ast.literal_eval(rm_modified)
    return result_material


result_material = process_material(file_path=file_path_materials)

result_ref = process_reflection(file_path_reflection)

result_thick = process_thickness(file_path_thickness)
