import csv


def write(filename, results) :
    with open(filename, 'wb') as csvfile :
        writer = csv.writer(csvfile)
        writer.writerow(['N', 'nIters', 'time'])
        for result in results :
            writer.writerow(result)



if __name__ == '__main__' :
    results = [
        (1, 2, 3),
        (4, 5, 6)
    ]
    write(results, 'test.csv')
    
