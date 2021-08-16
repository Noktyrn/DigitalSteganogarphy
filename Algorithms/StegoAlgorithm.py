class StegoAlgorithm(object):
    def __init__(self, insert_algo, extract_algo):
        self.inserter = insert_algo
        self.extractor = extract_algo
    
    def insert_message(self, im, mes):
        return self.inserter(message=mes, image=im)
    
    def extract_message(self, im, extract_params):
        return self.extractor(im, **extract_params)