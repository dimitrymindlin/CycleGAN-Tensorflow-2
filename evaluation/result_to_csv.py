import os
import csv


def create_json(input_string):
    data = {}
    lines = input_string.strip().split('\n')

    # Extract the starting time
    data['Name'] = lines[0].split()[1]

    # Extract the data for B2A and A2B
    for i in range(1, len(lines), 6):
        key = lines[i][2:].strip()  # remove the arrow and space at the beginning of the line
        values = {}
        values['TCV'] = float(lines[i + 1].split()[1])
        values['SSIM'] = float(lines[i + 2].split()[1])
        values['PSNR'] = float(lines[i + 3].split()[1])
        values['KID mean'] = float(lines[i + 4].split()[2])
        values['KID STD'] = float(lines[i + 5].split()[2])
        data[key] = values

    # Convert the data dictionary to a JSON object
    return data


def write_to_row(writer, data):
    # Write the data
    writer.writerow(
        [data['Name'],
         data['B2A']['TCV'],
         data['A2B']['TCV'],
         round((data['B2A']['TCV'] + data['A2B']['TCV']) / 2, ndigits=2),
         data['B2A']['KID mean'],
         data['A2B']['KID mean'],
         round((data['B2A']['KID mean'] + data['A2B']['KID mean']) / 2, ndigits=2),
         data['B2A']['SSIM'],
         data['A2B']['SSIM'],
         round((data['B2A']['SSIM'] + data['A2B']['SSIM']) / 2, ndigits=2),
         data['B2A']['PSNR'],
         data['A2B']['PSNR'],
         round((data['B2A']['PSNR'] + data['A2B']['PSNR']) / 2, ndigits=2),
         data['B2A']['KID STD'],
         data['A2B']['KID STD']
         ])


# List of file paths
dataset = "mura"
# experiments = ["2023-01-02--11.30", "2023-01-04--09.48", "2022-10-24--11.27"] #a2o
# experiments = ["2023-01-05--09.51", "2022-11-02--16.45", "2022-11-18--11.14"] #rsna
# experiments = ["2023-01-04--15.18", "2023-01-04--15.24", "2023-01-06--09.37"]  #h2z
experiments = ["2022-12-06--07.00", "2022-11-06--18.19", "2022-11-22--13.19", ]  # mura
file_names = ['test_metrics_16.txt', 'test_metrics_18.txt']

output_csv_name = f'../experiment_results_{dataset}.csv'
root_path = r"/Users/dimitrymindlin/UniProjects/CycleGAN-Tensorflow-2/checkpoints/gans/" + dataset

# Loop over the list of file paths
for experiment_path in experiments:
    # Navigate to the folder
    os.chdir(
        root_path + "/" + experiment_path)

    # Open the file
    for idx, file_name in enumerate(file_names):
        with open(file_name, 'r') as file:
            # Read the contents of the file into a string
            if idx == 0:
                file_180 = file.read()
            else:
                file_195 = file.read()

    input_files = [file_180, file_195]

    for input in input_files:
        # Create JSON data from string
        data = create_json(input)
        # Check if the file exists
        if os.path.exists(output_csv_name):
            # Open the CSV file in append mode
            with open(output_csv_name, 'a', newline='') as csvfile:
                # Create a CSV writer
                writer = csv.writer(csvfile)

                # Write the data
                write_to_row(writer, data)
        else:
            # Create a new CSV file
            with open(output_csv_name, 'w', newline='') as csvfile:
                # Create a CSV writer
                writer = csv.writer(csvfile)

                # Write the column names
                writer.writerow(
                    ['Name',
                     'TCV_B2A',
                     'TCV_A2B',
                     'TCV_Avg',
                     'KID_B2A',
                     'KID_A2B',
                     'KID_Avg',
                     'SSIM_B2A',
                     'SSIM_A2B',
                     'SSIM_Avg',
                     'PSNR_B2A',
                     'PSNR_A2B',
                     'PSNR_Avg',
                     'KID_STD_B2A',
                     'KID_STD_A2B'
                     ])

                # Write the data
                write_to_row(writer, data)
