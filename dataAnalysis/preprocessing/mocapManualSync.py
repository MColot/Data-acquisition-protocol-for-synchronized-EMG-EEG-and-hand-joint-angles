def offsetMocapData(pathToData, pathToOutput, offset, toMs=False):
    """
    Offsets motion capture data placed in a csv file by a given value and writes the result in a new csv file
    :param pathToData: path to file to modify
    :param pathToOutput: path to file where the result should be written
    :param timestamp: timestamp that marks the start of the recording to keep
    :param toMs: convert the timestamps from seconds to milli-seconds
    """
    resString = ""
    with open(pathToData, "r") as file:
        for line in file:
            data = line.split(";")
            dataTimestamp = float(data[0].replace(",", ".")) + offset
            if dataTimestamp >= 0:
                data[0] = dataTimestamp
                if toMs:
                    data[0] = round(1000 * dataTimestamp, 1)
                data[0] = str(data[0]).replace(",", ".")
                resString += ";".join(data)
    with open(pathToOutput, "w") as file:
        file.write(resString)



if __name__ == "__main__":

    offset = -(1635942056.905308 - 3600)
    foldername = "record_Dominique_03-11-21"
    name = "freemove6"
    inFile = f"C:/Users/marti/Desktop/memoire/data/{foldername}/{name}/{name}Quest.csv"
    outFile = f"C:/Users/marti/Desktop/memoire/data/{foldername}/{name}/{name}Quest_offset.csv"

    offsetMocapData(inFile, outFile, offset, True)
