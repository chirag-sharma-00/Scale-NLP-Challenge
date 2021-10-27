with open("data.txt", 'r') as datafile:
    data_len = len(datafile.readlines())
datafile.close()
with open("data.txt", 'r') as datafile:
    i = 0
    with open("val.txt", 'w') as valfile:
        with open("train.txt", 'w') as trainfile:
            while i < round(0.8 * data_len):
                line = datafile.readline()
                trainfile.write(line)
                i += 1
            while i < data_len:
                line = datafile.readline()
                valfile.write(line)
                i += 1
datafile.close()
trainfile.close()
valfile.close()