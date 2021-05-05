import csv
def wcsv(wfile, name):
    with open(str(name) + '.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='\n')
        writer.writerow(wfile)
