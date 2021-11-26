import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt


def read_csv(csv_file, flatten = True, verbose = False):
    """
    This Function takes CSV File as input, and returns numpy N-Dimensional Array.
    """
    # Read CSV File as DataFrame:
    df = pd.read_csv(csv_file, header=[0])

    label = df.iloc[:,0]
    data = df.iloc[:,1:]
    if verbose == True:
        print(f'File Head: {df.head()}')

    # Convert Data to Tensor:
    np_data = np.array(data)
    np_labels = np.array(label)

    x = torch.tensor(np_data)
    y = torch.tensor(label)

    if verbose == True:
        print(f'Shape of Data: {x.shape}')

    # Optional: Convert Data to 28 x 28: #TODO: Fix
    if flatten == False:
        x = x.view(-1, 1, 28, 28)
        # Verbose: Plot Image
        if verbose == True:
            print(f'Shape of Data (Reshaped): {x.shape}\n')
            print('Plotting Sample...')
            #plt.switch_backend('QtAgg4')
            plt.imshow(x[0][0])
            plt.show(block=False)
            input("press <ENTER> to continue")
              
    return x, y


def main_test():
    csv_file = 'C:/Users/igriffit/Documents/github-projects/Data-Science-Projects/Basic/MNIST Sign Language/Data/sign_mnist_train.csv'
    read_csv(csv_file, flatten = False, verbose = True)
    pass


if __name__ == '__main__':
    main_test()
    #dev()