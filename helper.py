import os

# Number of samples div batch size
def stepCount(directory, batch_size):
    return samplesInDirectory(directory) // batch_size

def samplesInDirectory(directory):
    count = 0;
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".jpg"):
                count += 1
    return count

