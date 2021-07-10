import ir_algorithms

file_name = 'data\\IR00_3_11k News.xlsx'
num_clusters = 20

def main():
    while(True):
        print('1: Index documents from scratch')
        print('2: Run queries using the existing Inverted Index')
        print('3: Run k-means and build clusters')
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
            ir_system.cluster(num_clusters)
        print('\ndone\n')



if __name__ == '__main__':
    main()