import openpyxl # for reading excel files
import re, regex, pickle, numpy as np
import heapq, math, random

frequent_terms_num = 20 # removing # of most frequent terms from dictionary
max_results_num = 20 # maximum number of results to show
champions_list_size = 100

ranked_retrieval = True # whether or not to use ranked retrieval
use_index_elimination = True # whether or not to use index elimination technique
use_champions_list = True # whether or not to use champions lists technique
use_heap = True # whether or not to use max-heap in order to return top_k results

arabic_plurals_file = 'arabic_plurals.txt'
verbs_stems_file = 'verbs_stems.txt'

arabic_persian_chars = [['ي', 'ی'], ['ئ', 'ی'], ['ك', 'ک'], ['ة', 'ه'], ['ؤ', 'و'],\
             ['آ', 'ا'], ['إ', 'ا'], ['أ', 'ا'], ['ٱ', 'ا'], ['ء', '']]

# end_words = ['ان', 'ات', 'تر', 'تری', 'ترین', 'م', 'ت', 'ش', 'یی', 'ی', 'ها', 'ا']
end_words = ['ات', 'تر', 'تری', 'ترین', 'یی', 'ی', 'ها']

prefixes = ['ابر', 'باز', 'پاد', 'پارا', 'پسا', 'پیرا', 'ترا', 'فرا', 'هم', 'فرو']
postfixes = ['اسا', 'آگین', 'گین', 'ومند', 'اک', 'اله', 'انه', 'ین'\
    'ینه', 'دان', 'کار' , 'دیس', 'زار', 'سار', 'ستان', 'سرا', 'فام', 'کده', 'گار', \
        'گان', 'گری', 'گر', 'گون', 'لاخ', 'مان', 'مند', 'ناک', 'نده', 'وار', 'واره',\
            'واری', 'ور', 'وش']

past_verb_post = ['م', 'ی', '' , 'یم', 'ید', 'ند']


# Print iterations progress
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

class Document:
    def __init__(self, doc_id, content, topic, url) -> None:
        self.doc_id = doc_id
        self.content = content
        self.topic = topic
        self.url = url

    def __str__(self) -> str:
        return 'doc_id: ' + str(self.doc_id) + '\ttopic: ' + str(self.topic) + '\turl: ' + str(self.url)

    def __repr__(self) -> str:
        return str(self)

class Posting:
    def __init__(self, doc_id, freq) -> None:
        self.doc_id = doc_id
        self.freq = freq
        self.weight = None

    def __str__(self) -> str:
        return 'doc_id: ' + str(self.doc_id) + '\tfreq: ' + str(self.freq) + '\tweight: ' + str(self.weight)

    def __repr__(self) -> str:
        return str(self)


class PostingsList:
    def __init__(self) -> None:
        self.plist = []
        self.term_freq = 0
        self.champs_list = []

    def __str__(self) -> str:
        return 'term_freq: ' + str(self.term_freq) + '\t' + str(self.plist)

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other):
        return self.term_freq < other.term_freq

    
    


class IR:
    def __init__(self) -> None:
        self.dictionary = dict()
        self.documents = None
        self.docs_dict = dict()
        self.arabic_plurals_dict = dict()
        self.verbs_dict = dict()
        self.docs_vectors = dict()

    
    # building the inverted index
    def build_inverted_index(self, file_name):
        global arabic_plurals_file, verbs_stems_file
        self.init_file(file_name)
        # initializing arabic_plurals stemming dictionary
        self.init_arabic_plurals(arabic_plurals_file)
        # initializing verb to verb-stems dictionary
        self.init_verbs_dict(verbs_stems_file)
        indexed_docs_num = 0
        print('Indexing documents...')
        for doc in self.documents:
            print_progress_bar(indexed_docs_num/len(self.documents), 1, prefix = 'Progress:', suffix = 'Complete', length = 50)
            self.index_document(doc)
            indexed_docs_num += 1
        print('Inverted Index Matrix construction completed')
        self.build_docs_dict()
        # removing most frequent items
        self.remove_frequents(frequent_terms_num)
        #calculating tf-idf weights for each posting
        print('Updating Posting Weights...')
        self.update_postings_weights()
        print('Posting Weights Updated!')
        # building champions lists for each term
        print('Building Champions Lists...')
        self.build_champions_lists()
        print('Champions Lists have been built!')
        # building document vectors representations
        print('Building vector-space representations of documents...')
        self.build_doc_vectors()
        print('Vector-space representations of documents have been built!')
        # saving the dictionary
        with open('data\\index.pickle', 'wb') as handle:
            pickle.dump(self.dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data\\docs_dict.pickle', 'wb') as handle:
            pickle.dump(self.docs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data\\arabic_plurals_dict.pickle', 'wb') as handle:
            pickle.dump(self.arabic_plurals_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data\\verbs_stems_dict.pickle', 'wb') as handle:
            pickle.dump(self.verbs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('data\\docs_vectors.pickle', 'wb') as handle:
            pickle.dump(self.docs_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # building a dictionary mapping documents IDs to URL
    def build_docs_dict(self):
        for doc in self.documents:
            self.docs_dict[doc.doc_id] = doc.url


    
    # loading the existing inverted index from file
    def load_inverted_index(self):
        with open('data\\index.pickle', 'rb') as handle:
            self.dictionary = pickle.load(handle)
        with open('data\\docs_dict.pickle', 'rb') as handle:
            self.docs_dict = pickle.load(handle)
        with open('data\\arabic_plurals_dict.pickle', 'rb') as handle:
            self.arabic_plurals_dict = pickle.load(handle)
        with open('data\\verbs_stems_dict.pickle', 'rb') as handle:
            self.verbs_dict = pickle.load(handle)
        with open('data\\docs_vectors.pickle', 'rb') as handle:
            self.docs_vectors = pickle.load(handle)



    # initializing documents list by reading the excel dataset
    def init_file(self, file_name):
        print('Initializing documents using Excel file...')
        wb_obj = openpyxl.load_workbook(file_name)
        sheet = wb_obj.active
        headers = []
        dataset = []
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i == 0:
                for header in row:
                    headers.append(header)
            else:
                document = Document(row[0], row[1], row[2], row[3])
                dataset.append(document)
        self.documents = dataset
        print('Initialized Excel file')


    # processing the documents one by one for building the index
    def index_document(self, doc):
        tokens = self.get_tokens(doc.content)
        counts = self.get_counts_dict(tokens)
        unique_tokens = counts.keys()
        for unique_token in unique_tokens:
            posting = Posting(doc.doc_id, counts[unique_token])
            self.add_posting(posting, unique_token)

    
    # get tokens for each document
    def get_tokens(self, text):
        persian_letters = r'[آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئؤأإةيكء]+'
        '''
        tokens = re.split('!|,|[|]|\{|\}|\s|-|_|\(|\)|\.|؟|:|»|«|\(|\)|؛|،|\*|&|\
            \^|%|\$|#|@|~|\\|\"|"|\'|;|>|<|\||=|\+|\?', text)
        tokens = list(filter(None, tokens))
        '''
        # tokens = regex.findall(r'[\p{Cf}\p{L}]+', text)
        tokens = re.findall(persian_letters,text)
        # modifying tokens
        # tokens2 = [self.modify_token(token) for token in tokens if self.modify_token(token) != '']
        i = 0
        for token in tokens[:]:
            modified_token = self.modify_token(token)
            if len(modified_token) > 1 and modified_token != '':
                tokens[i] = modified_token
            else:
                del(tokens[i])
                i -= 1
            i += 1
        return tokens

    
    # getting a dictionary of unique terms and the frequencies of each term in a list
    def get_counts_dict(self, tokens):
        counts = dict()
        for token in tokens:
            if token not in counts:
                counts[token] = 1
            else:
                counts[token] += 1
        return counts


    # add posting to the postings_list of the corresponding term in dictionary
    def add_posting(self, posting, term):
        postings_list = None
        if term not in self.dictionary:
            postings_list = PostingsList()
        else:
            postings_list = self.dictionary[term]
        postings_list.term_freq += 1
        postings_list.plist.append(posting)
        self.dictionary[term] = postings_list
        
            

    
    # modify token with stemming, tokenization, normalization, etc
    def modify_token(self, token):
        # to be implemented...
        #
        #
        #
        # m = re.findall(r"^ب[ا-ی]*ید$", token)
        # if m:
        #     if m[0] not in unique_list:
        #         unique_list.append(m[0])
        # removing stop words
        #if len(token) < 3:
        #    return ''
        token = self.normalize(token)
        token = self.stem(token)
        return token


    # normalize the given token
    def normalize(self, token):
        # normalizing characters
        result = token
        # list of [Arabic_character, Persian_character]
        global arabic_persian_chars
        for i in range(len(arabic_persian_chars)):
            if arabic_persian_chars[i][0] in token:
                result = result.replace(arabic_persian_chars[i][0], arabic_persian_chars[i][1])
        '''
        for char_set in arabic_persian_chars:
            result = re.sub(char_set[0], char_set[1], result)
        '''
        return result

    # stemming words and verbs
    def stem(self, token):
        # stemming verbs
        if token in self.verbs_dict:
            token = self.verbs_dict[token]
            return token
        # removing postfixes
        for end in end_words:
            if token.endswith(end):
                token = token[:-len(end)]
        for post in postfixes:
            if token.endswith(post):
                token = token[:-len(post)]
        for pre in prefixes:
            if token.startswith(pre):
                token = token[len(pre):]
        # stemming arabic plurals
        if token in self.arabic_plurals_dict:
            token = self.arabic_plurals_dict[token]
        return token


    # processing queries
    def process_query(self, query):
        tokens = self.get_tokens(query)
        if ranked_retrieval:
            self.process_ranked_query(tokens)
            return
        if len(tokens) == 1:
            self.process_query_single_word(tokens[0])
        elif len(tokens) > 1:
            self.process_query_mult_words(tokens)
        else:
            print('Query is not valid; please make sure your query contains words.')
    

    # processing queries with only one word
    def process_query_single_word(self, query):
        result_ids = self.get_posting_ids(query)
        if len(result_ids) == 0:
            print("No result found!")
        else:
            showed_result_count = 0
            print('Found ' + str(len(result_ids)) + ' Results: (Showing at max ' + str(max_results_num) + ')')
            for index in result_ids:
                if showed_result_count >= max_results_num:
                    break
                print(index, self.docs_dict[index])
                showed_result_count += 1
        return result_ids


    # processing queries with multiple words - alternative solution using intersection
    def process_query_mult_words_alt(self, tokens):
        id_lists = []
        pointers = []
        result_set = []
        for t in tokens:
            posting_ids = self.get_posting_ids(t)
            id_lists.append(posting_ids)
            if len(posting_ids) > 0:
                pointers.append(0)
            else:
                pointers.append(None)
        terminated = all(x is None for x in pointers)
        while not terminated:
            min_index = len(self.documents) + 1
            min_pointer_ind = len(pointers)
            for i in range(len(pointers)):
                if pointers[i] is None:
                    continue
                id_list = id_lists[i]
                pointer = pointers[i]
                if id_list[pointer] < min_index:
                    min_index = id_list[pointer]
                    min_pointer_ind = i
            # found minimum
            if result_set == [] or result_set[len(result_set)-1][0] != min_index:
                result_set.append([min_index, 1])
            else:
                result_set[len(result_set)-1][1] += 1
            # move min_pointer one step further
            pointers[min_pointer_ind] += 1
            if pointers[min_pointer_ind] >= len(id_lists[min_pointer_ind]):
                pointers[min_pointer_ind] = None
            terminated = all(x is None for x in pointers)
        result_set = sorted(result_set, key=lambda item: item[1], reverse=True)
        return result_set



    # processing queries with multiple words
    def process_query_mult_words(self, tokens):
        id_lists = []
        result_set = dict()
        for t in tokens:
            posting_ids = self.get_posting_ids(t)
            id_lists.append(posting_ids)
        for id_list in id_lists:
            for item in id_list:
                if item not in result_set:
                    result_set[item] = 1
                else:
                    result_set[item] += 1
        result_set = sorted(result_set.items(), key=lambda item: (-item[1], item[0]))
        last_count = 0
        if len(result_set) == 0:
            print("No result found!")
        else:
            print('Found ' + str(len(result_set)) + ' Results: (Showing at max ' + str(max_results_num) + ')')
            showed_results_num = 0
            for item in result_set:
                if showed_results_num >= max_results_num:
                    break
                if last_count != item[1]:
                    print('\nNumber of Words:', item[1])
                    last_count = item[1]
                index = item[0]
                print(index, self.docs_dict[index])
                showed_results_num += 1
        return result_set

    
    # Processing queries using rank-based method
    def process_ranked_query(self, query_tokens):
        query_terms_temp = list(set(query_tokens))
        query_terms = []
        for term in query_terms_temp:
            if term in self.dictionary:
                query_terms.append(term)
        query_terms_freqs = [query_tokens.count(term) for term in query_terms]
        query_terms_weights = []
        # calculating query-terms weights
        for i in range(len(query_terms)):
            term = query_terms[i]
            term_freq = query_terms_freqs[i]
            doc_freq = self.dictionary[term].term_freq
            weight = self.calculate_tf_idf(term_freq, doc_freq, len(self.docs_dict))
            query_terms_weights.append(weight)
        # normalizing query weights
        query_len = sum([weight**2 for weight in query_terms_weights])
        query_len = math.sqrt(query_len)
        query_terms_norm_weights = [weight/query_len for weight in query_terms_weights]
        # calculating each doc's scores
        # scores = [0] * (len(self.docs_dict) + 1)
        scores = dict()
        # doc_lens = [0] * (len(self.docs_dict) + 1)
        docs_lens = dict()
        for term in query_terms:
            posting_ids = None
            if use_champions_list:
                posting_ids = self.get_champs_ids(term)
            else:
                posting_ids = self.get_posting_ids(term)
            for posting_id in posting_ids:
                scores[posting_id] = 0
                docs_lens[posting_id] = 0
        for i in range(len(query_terms)):
            query_term = query_terms[i]
            w_tq = query_terms_norm_weights[i]
            plist = None
            if use_champions_list:
                plist = self.dictionary[query_term].champs_list
            else:
                plist = self.dictionary[query_term].plist
            for posting in plist:
                doc_id, w_td = posting.doc_id, posting.weight
                scores[doc_id] += w_td * w_tq
                docs_lens[doc_id] += w_td**2
        for key in scores.keys():
            if docs_lens[key] != 0:
                docs_lens[key] = math.sqrt(docs_lens[key])
                # normalizing doc-weights vectors by their len in score
                scores[key] /= docs_lens[key]
        top_k = self.retrieve_top_k(scores, max_results_num)
        self.show_results(top_k)
        return top_k

            
    # getting top_k highest cosine scores
    def retrieve_top_k(self, scores, k):
        top_k = []
        if use_heap:
            heap = []
            # building the heap (actually a min heap with -score equivalent max heap with score)
            for item in scores.items():
                heapq.heappush(heap, (-item[1], item[0]))
            # getting k highest scores (equivalent to k smallest -scores)
            for i in range(k):
                if len(heap) == 0:
                    break
                item = heapq.heappop(heap)
                top_k.append([item[1], -item[0]])
        else:
            scores = list(scores.items())
            scores = sorted(scores, key=lambda item: item[1], reverse=True)
            if len(scores) <= k:
                top_k = scores
            else:
                top_k = scores[:k]
        return top_k

    
    # printing query results for the user
    def show_results(self, results):
        result_ids = [res[0] for res in results]
        if len(results) == 0:
            print("No result found!")
        else:
            print('Results: (Showing top ' + str(max_results_num) + ')')
            for index in result_ids:
                print(index, self.docs_dict[index])




    # getting document IDs for a given term
    def get_posting_ids(self, term):
        ids = []
        if term in self.dictionary:
            postings_list = self.dictionary[term]
            postings = postings_list.plist
            for p in postings:
                ids.append(p.doc_id)
        return ids

    
    # getting champions-lists posting ids
    def get_champs_ids(self, term):
        ids = []
        if term in self.dictionary:
            postings_list = self.dictionary[term]
            postings = postings_list.champs_list
            for p in postings:
                ids.append(p.doc_id)
        return ids


    # updating tf-idf weights for every posting
    def update_postings_weights(self):
        for key in self.dictionary.keys():
            plist, doc_freq = self.dictionary[key].plist, self.dictionary[key].term_freq
            for posting in plist:
                term_freq = posting.freq
                posting.weight = self.calculate_tf_idf(term_freq, doc_freq, len(self.docs_dict))


    # building champions lists weights
    def build_champions_lists(self):
        for term in self.dictionary.keys():
            plist = self.dictionary[term].plist
            champs_list = self.dictionary[term].champs_list
            top_postings = heapq.nlargest(champions_list_size, plist, key=lambda p: p.weight)
            top_postings = sorted(top_postings, key=lambda p: p.doc_id)
            for posting in top_postings:
                champs_list.append(posting)

    
    # updating tf-idf weights for a single posting
    def calculate_tf_idf(self, term_freq, doc_freq, num_docs):
        tf_weight = 1 + math.log10(term_freq)
        idf_weight = math.log10(num_docs/doc_freq)
        tf_idf = tf_weight * idf_weight
        return tf_idf
            


    # removing k most frequent term from dictionary
    def remove_frequents(self, k):
        most_frequent = heapq.nlargest(k, self.dictionary, key=self.dictionary.get)
        for key in most_frequent:
            del self.dictionary[key]

    
    # initializing arabic_plurals_dict
    def init_arabic_plurals(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as reader:
            for line in reader:
                words = line.split()
                (singular, plural) = (self.normalize(words[0]), self.normalize(words[1]))
                self.arabic_plurals_dict[plural] = singular


    # generating all verb tenses
    def generate_verb_tenses(self, verb_root):
        # past verbs
        tenses = set()
        past_verb_posts = ['م', 'ی', '' , 'یم', 'ید', 'ند']
        past_root, present_root = verb_root.split(r'$')
        (present_root, past_root) = (self.normalize(present_root), self.normalize(past_root))
        if present_root == 'هست':
            for post in past_verb_posts:
                tenses.add(present_root + post)
                tenses.add('نیست' + post)
            return tenses
        for post in past_verb_posts:
            tenses.add(past_root + post)
            tenses.add('ن' + past_root + post)
            tenses.add('می' + past_root + post)
            tenses.add('نمی' + past_root + post)
        # present verbs
        tenses.add(present_root)
        present_verb_posts = ['م', 'ی', 'د', 'یم', 'ید', 'ند']
        if present_root == 'ا' or present_root == 'گو' or present_root.endswith('ا'):
            present_root = present_root + 'ی'
        for post in present_verb_posts:
            tenses.add(present_root + post)
            tenses.add('می' + present_root + post)
            tenses.add('نمی' + present_root + post)
            # imperatives
            tenses.add('ب' + present_root + post)
            tenses.add('ن' + present_root + post)
        # Masdar
        tenses.add(past_root + 'ن')
        # perfect present
        tenses.add(past_root + 'ه')
        return tenses


    # building dictionary for stemming verbs
    def init_verbs_dict(self, file_name):
        with open(file_name, 'r', encoding='utf-8') as reader:
            for line in reader:
                verb_root = line.split()[0]
                tenses = self.generate_verb_tenses(verb_root)
                for tense in tenses:
                    self.verbs_dict[tense] = verb_root         


    # represent document content as a vector
    def get_doc_vec(self, content):
        tokens = self.get_tokens(content)
        temp_terms = list(set(tokens))
        terms = []
        for term in temp_terms:
            if term in self.dictionary:
                terms.append(term)
        terms_freqs = [tokens.count(term) for term in terms]
        terms_weights = []
        # calculating query-terms weights
        for i in range(len(terms)):
            term = terms[i]
            term_freq = terms_freqs[i]
            doc_freq = self.dictionary[term].term_freq
            weight = self.calculate_tf_idf(term_freq, doc_freq, len(self.docs_dict))
            terms_weights.append(weight)
        # normalizing query weights
        doc_len = sum([weight**2 for weight in terms_weights])
        doc_len = math.sqrt(doc_len)
        terms_norm_weights = [weight/doc_len for weight in terms_weights]
        doc_vec = dict(zip(terms, terms_norm_weights))
        return doc_vec

    # Building vector-space representations of documents
    def build_doc_vectors(self):
        indexed_num = 0
        for doc in self.documents:
            vec = self.get_doc_vec(doc.content)
            self.docs_vectors[doc.doc_id] = vec
            indexed_num += 1
            print_progress_bar(indexed_num/len(self.documents), 1, prefix = 'Progress:', suffix = 'Complete', length = 50)

    
    # calculating cosine-similarity between two document vectors
    def cosine_score(self, vec1, vec2):
        score = 0
        base_vec, addit_vec = vec1, vec2
        if len(vec1) > len(vec2):
            base_vec, addit_vec = vec2, vec1
        for term in base_vec.keys():
            if term in addit_vec:
                score += base_vec[term] * addit_vec[term]
        return score

    
    # fiding best (= nearest) centroid for a document
    def find_best_centroid(self, vec, centroids):
        best_ind = 0
        best_score = 0
        for i in range(len(centroids)):
            score = self.cosine_score(vec, centroids[i])
            if score > best_score:
                best_score = score
                best_ind = i
        return best_ind


    # building clusters using k-means algorithm
    def cluster(self, k):
        max_iter = 1000
        seeds_inds = random.sample(range(1, len(self.docs_vectors) + 1), k)
        centroids = [self.docs_vectors[ind] for ind in seeds_inds]
        clusters = [dict() for i in range(k)]
        for iter in range(max_iter):
            for i in range(1, len(self.docs_vectors) + 1):
                vec = self.docs_vectors[i]
                label = self.find_best_centroid(vec, centroids)
                clusters[label][i] = vec
            print('{} out of {}'.format(iter, max_iter))
        
    
        