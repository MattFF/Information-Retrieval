import os
import math
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

class BooleanModel:
    def __init__(self, terms_):
        self.documents = []
        self.docs_processed = []
        self.index_inverted = {}
        self.terms = terms_

    def collect(self):
        '''
            A função acessa a pasta principal (20 News Groups), 
            onde para cada sub-pasta vão ser recuperados os documentos.
        '''
        print('Acessando o diretório "20 News groups"...\n')
        if os.path.isdir('20 News groups'):
            os.chdir('20 News groups')
        else:
            print('O diretorio não foi encontrado.')
            return

        with os.scandir() as directories:
            for directory in directories:
                self.documents += [[doc, 0.0] for doc in os.scandir(directory)]
                    

    def capitalization(self, tokens):
        return [
            token.lower() for token in tokens
            if token.isalnum()
        ]


    def stop_words(self, tokens):
        stop_words = stopwords.words('english')
        return [
            token for token in tokens
            if token not in stop_words
        ]


    def lemmatization(self, tokens):
        lemmatizer = WordNetLemmatizer()
        return [
            lemmatizer.lemmatize(token, 'n') for token in tokens
        ]


    def stemming(self, tokens):
        ps = PorterStemmer()
        return [
            ps.stem(token) for token in tokens
        ]


    def weight_log(self, weight):
        if weight == 0:
            return 0
        else:
            return 1 + math.log10(weight)


    def tf_idf(self):
        self.terms = self.pre_processing(self.terms)
        for index, tokens in enumerate(self.docs_processed):
            score = 0.0
            for term in self.terms:
                tf_idf = self.tf(tokens, term) * self.idf(term)
                score += tf_idf
            self.documents[index][1] = score


    def idf(self, term):
        if self.index_inverted.get(term, 0):
            return math.log10(len(self.documents)
                              / len(self.index_inverted[term]['postings']))
        else:
            return 0


    def tf(self, tokens, term):
            text = " ".join(tokens)
            return self.weight_log(text.count(term))


    def pre_processing(self, tokens):
        tokens = self.capitalization(tokens)
        tokens = self.stop_words(tokens)
        tokens = self.lemmatization(tokens)
        return tokens


    def inverted_index(self, tokens, id_):
        for token in tokens:
            if token not in self.index_inverted:
                self.index_inverted[token] = {
                    'postings': {id_ + 1},
                }
            else:
                self.index_inverted[token]['postings'].add(id_ + 1)


    def prettyprint(self, results):
        print('Resultados:\n')
        for result in results[:10]:
            print(f'Documento - {result[0].name}')


    def tokenize(self):
        tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
        for id_, document in enumerate(self.documents):
            with open(document[0].path, 'r', errors='ignore') as doc:
                lines = " ".join(doc.readlines())
                tokens_ = tokenizer.tokenize(lines.replace('\n', ' '))
                tokens_ = self.pre_processing(tokens_)
                self.docs_processed.append(tokens_)
                self.inverted_index(tokens_, id_)
        print(f'Tamanho do indice invertido: {len(self.index_inverted)}')

    def run(self):
        self.collect()
        self.tokenize()
        self.tf_idf()
        self.prettyprint(sorted(self.documents, key=lambda x: x[1], reverse=True))


query = input("Buscar: ")
boolean = BooleanModel(query.split(' '))
boolean.run()