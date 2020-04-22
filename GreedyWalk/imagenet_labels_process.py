import os
import numpy as np

def main():
    # Create labels vector for validation data 1-10,000
    fileNameToID = np.load('fileNameToID.npy', allow_pickle=True).item()
    image_dir = './imagenet_sample_data'

    idToLabel = np.load('synset_map.npy', allow_pickle=True).item()

    fns = os.listdir(image_dir)
    fns.sort()

    output = np.zeros((10000, 1))
    for i, fn in enumerate(fns):
        fn = os.path.splitext(fn)[0]
        print(fn, fileNameToID[fn], idToLabel[fileNameToID[fn]])
        output[i] = idToLabel[fileNameToID[fn]]
    
    nparray = np.asarray(output)

    print(nparray)
    # print(nparray.shape)

    np.save('val_labels_10000.npy', nparray)

if __name__ == '__main__':
    main()