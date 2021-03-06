import ir_algorithms
import time
from os.path import exists

# file_name = 'data\\IR00_3_11k News.xlsx'
file_name = 'data\\Merged_50k_News.xlsx'
unlabeled_dataset = 'data\\IR_Spring2021_ph12_7k.xlsx'
file_list = ['data\\IR00_3_11k News.xlsx', 'data\\IR00_3_17k News.xlsx', 'data\\IR00_3_20k News.xlsx']
num_clusters = 20
num_inits = 5
b2_param = 4

def main():
    while(True):
        print('1: Index documents from scratch')
        print('2: Run queries using the existing Inverted Index')
        print('3: Run k-means and build clusters')
        print('4: Run queries using clustering technique')
        print('5: Build classification model and classify unlabeled docs')
        print('6: Build train and test vectors from excel files')
        print('7: Run queries using classification technique')
        option = input('Select option: ')
        ir_system = ir_algorithms.IR()
        if not exists(file_name):
            ir_system.merge_documents(file_list, file_name)
        if option == '1':
            ir_system.build_inverted_index(file_name)
        elif option == '2':
            ir_system.load_inverted_index()
            query = input("Enter your query: ")
            start_time = time.time()
            ir_system.process_query(query)
            end_time = time.time()
            print('\nTotal Time:', end_time-start_time, 'seconds')
        elif option == '3':
            ir_system.load_inverted_index()
            ir_system.cluster(num_clusters, num_inits)
        elif option == '4':
            ir_system.load_inverted_index()
            ir_system.load_cluster_data()
            query = input("Enter your query: ")
            start_time = time.time()
            ir_system.process_query_using_clustering(query, b2_param)
            end_time = time.time()
            print('\nTotal Time:', end_time-start_time, 'seconds')
        elif option == '5':
            ir_system.load_inverted_index()
            ir_system.classify()
        elif option == '6':
            ir_system.load_inverted_index()
            ir_system.build_classification_vectors(file_name, unlabeled_dataset)
        elif option == '7':
            ir_system.load_inverted_index()
            query = input("Enter your query: ")
            start_time = time.time()
            ir_system.process_query_using_classification(query)
            end_time = time.time()
            print('\nTotal Time:', end_time-start_time, 'seconds')
        print('\ndone\n')



if __name__ == '__main__':
    main()