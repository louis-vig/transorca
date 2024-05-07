import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_losses = []
    corrs = []
    validation_losses = []
    with open("training_data.txt") as f:
        line = f.readline()
        while line:
            arr = line.split()
            if arr[0] == "|" or arr[0] == "\n":
                line = f.readline()
                continue
            elif arr[1] == "Loss":
                training_losses.append(float(arr[2]))
            elif arr[1] == "Corr" and arr[3] == "MSE":
                corrs.append(float(arr[2]))
                validation_losses.append(float(arr[4]))
            else:
                raise ValueError(f"Unexpected value encountered while parsing: {line}")
            line = f.readline()
    
    training_loss_checkpoints = list(range(0, 500*len(training_losses), 500))
    validation_checkpoints = list(range(2000, 2000*(len(validation_losses)+1), 2000))

    plt.subplot(121)
    plt.plot(training_loss_checkpoints[1:], training_losses[1:], "r-")
    plt.title("Training Loss")
    plt.xlabel("Training iteration")
    plt.ylabel("Mean squared error")
    plt.subplot(122)
    plt.plot(validation_checkpoints, validation_losses, 'b-')
    plt.title("Validation Loss")
    plt.xlabel("Training iteration")
    plt.ylabel("Mean squared error")
    plt.savefig('foo.pdf')
    plt.close()

    plt.plot(validation_checkpoints, corrs, 'g-')
    plt.xlabel("Training iteration")
    plt.ylabel("Average correlation")
    plt.savefig('bar.pdf')
