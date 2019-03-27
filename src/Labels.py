import csv


class Labels():
    def __init__(self):
        self.current_lng = 0

        self.texts = []
        self.txt = []

        self.load_csv()
        self.change_language(self.current_lng)

    def load_csv(self):
        text_file = open('src/lng_file.csv', 'rt', encoding='utf-8')

        csv_reader = csv.reader(text_file, delimiter=";")

        for row in csv_reader:
            self.texts.append(row)

    def change_language(self, lng):
        self.current_lng = lng
        self.txt = []
        for line in self.texts:
            self.txt.append(line[self.current_lng])
