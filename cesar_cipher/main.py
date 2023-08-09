import numpy as np 
from Datater import Datater
from tensorflow import keras


def main():
    dttr = Datater(shift=2)
    dttr.generate_dataset("test", 300, 10000)

    PREF = "/home/anton/nn/cesar_cipher/datasets/"

    input_data = []
    output_data = []
    with open(PREF + "test" + "_inputs.txt", "r") as x_file:
        lines = x_file.readlines()
        maxlen = max([len(line[:-1]) for line in lines])
        for line in lines: 
            input_data.append(dttr.string_to_utf_vec(line[:-1]) + [0]*(maxlen-len(line[:-1])) )
    with open(PREF + "test" + "_outputs.txt", "r") as y_file:
        lines = y_file.readlines()
        maxlen = max([len(line[:-1]) for line in lines])
        for line in lines: 
            output_data.append(dttr.string_to_utf_vec(line[:-1]) + [0]*(maxlen-len(line[:-1])))

    N = len(input_data)

    x_train = np.array(input_data[:int(1.0 * N):])
    y_train = np.array(output_data[:int(1.0 * N):])
    # x_test = np.array(input_data[int(0.90 * N):])
    # y_test = np.array(output_data[int(0.90 * N):])

    model = keras.Sequential([
        keras.layers.Input(maxlen),
        keras.layers.Dense(maxlen),
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=keras.losses.mean_squared_error)
    model.fit(x_train, y_train, epochs=1500, verbose=2, batch_size=500)

    msg = "BLYAT' FUCK FUCK FUCK FUCK FUCKKKK"
    print(f"\nMessage for encrypt: {msg}\n")
    print(f"Encrypted message: {dttr.encrypt(msg, mode=Datater.modes.string)}\n")
    input = np.array(dttr.encrypt(msg, mode=Datater.modes.utf) + [dttr.shift]*(maxlen-len(msg)))
    print(f"\nDecrypted message: {dttr.utf_vec_to_string(list(model.predict(input[np.newaxis, :])[0]))} \n")

    # print("\n\n\n")
    # print(dttr.string_to_utf_vec(msg))
    # print(list(model.predict(input[np.newaxis, :])[0]))


if __name__ == "__main__":
    main()