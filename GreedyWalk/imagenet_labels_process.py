import os
import numpy as np

def main():
    # Create labels vector for validation data 1-10,000
    fileNameToID = np.load('fileNameToID.npy', allow_pickle=True)
    image_dir = './val_subset'
    fns = os.listdir(image_dir)
    fns.sort()
    output = np.zeros((10000, 1))
    for i, fn in enumerate(fns):
        output[i] = fileNameToID[fn]
    
    nparray = np.asarray(output)

    np.save('val_labels_10000.npy', nparray)

if __name__ == '__main__':
    main()