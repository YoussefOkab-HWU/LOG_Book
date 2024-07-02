import subprocess

def log_terminal_command(command, log_file):
    try:
        with open(log_file, 'a') as f:
            f.write(f"Command: {command}\n")
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in process.stdout:
                f.write(line)
                print(line, end='')  # Print the output in real-time
            for line in process.stderr:
                f.write(line)
                print(line, end='')  # Print the error in real-time
            print(f"Command '{command}' executed successfully. Output logged to '{log_file}'")
    except Exception as e:
        print(f"Error executing command '{command}': {e}")

if __name__ == "__main__":
    command = "python3 nifi_yolo_soil_extraction_version_3.py"
    log_file = "soil_analysis.log"
    log_terminal_command(command, log_file)