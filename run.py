
import subprocess

# Run a bash file
result = subprocess.run(['/bin/bash', 'run.sh'])

# Print the output
print(result.stdout)
