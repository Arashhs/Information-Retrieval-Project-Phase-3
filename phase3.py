import ir_algorithms

file_name = 'data\\IR00_3_11k News.xlsx'
unlabeled_dataset = 'data\\IR_Spring2021_ph12_7k.xlsx'
num_clusters = 30
num_inits = 5
b2_param = 5

def main():
    while(True):
        print('1: Index documents from scratch')
        print('2: Run queries using the existing Inverted Index')
        print('3: Run k-means and build clusters')
        print('4: Run queries using clustering technique')
        print('5: Build classification model and classify unlabeled docs')
        option = input('Select option: ')
        ir_system = ir_algorithms.IR()
        if option == '1':
            ir_system.build_inverted_index(file_name)
        elif option == '2':
            ir_system.load_inverted_index()
            query = input("Enter your query: ")
            ir_system.process_query(query)
        elif option == '3':
            ir_system.load_inverted_index()
            ir_system.cluster(num_clusters, num_inits)
        elif option == '4':
            ir_system.load_inverted_index()
            ir_system.load_cluster_data()
            query = input("Enter your query: ")
            ir_system.process_query_using_clustering(query, b2_param)
        elif option == '5':
            ir_system.load_inverted_index()
            ir_system.classify(file_name, unlabeled_dataset)
        print('\ndone\n')



if __name__ == '__main__':
    main()