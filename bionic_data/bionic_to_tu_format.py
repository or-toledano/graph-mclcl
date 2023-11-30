
import glob
import pickle
import os

def create_dataset_dirs(dataset_name):
    paths = [f'data/{dataset_name}', f'data/{dataset_name}/{dataset_name}',
             f'data/{dataset_name}/{dataset_name}/raw', f'data/{dataset_name}/{dataset_name}/processed']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def main():
    datasets_path = "bionic_data"
    datasets = glob.glob(f'{datasets_path}/*.txt')

    for file in datasets:
        dataset_name = file.split('.')[0].split('\\')[1]
        node_counter = 1
        node_dict = dict()

        def get_node_id(node_name):
            nonlocal node_counter
            if node_name not in node_dict:
                node_counter += 1
                node_dict[node_name] = node_counter
            return node_dict[node_name]

        with open(file, 'r') as f:
            lines = f.readlines()
        new_file_str = ""
        for line in lines:
            split = line.split(' ')
            if len(split) == 3:
                node_1, node_2, weight = line.split(' ')
            elif len(split) == 2:
                node_1, node_2 = line.split(' ')
                weight = "1.0\n"
            node_1_id = get_node_id(node_1)
            node_2_id = get_node_id(node_2)
            #new_file_str += f"{node_1_id},{node_2_id},{weight}"
            new_file_str += f"{node_1_id},{node_2_id}\n"
        raw_path = f'data/{dataset_name}/{dataset_name}/raw'
        with open(f'{raw_path}/node_dict_{dataset_name}.pickle', 'wb') as handle:
            pickle.dump(node_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Create the download file paths:
        create_dataset_dirs(dataset_name)

        with open(f'{raw_path}/{dataset_name}_A.txt', 'w') as new_f:
            new_f.write(new_file_str)
        # if need to open the node_dict later use this:
        # with open(f'node_dict_{dataset_name}.pickle', 'rb') as handle:
        #     b = pickle.load(handle)

        # write the graph indicator file:
        graph_indicator_str = ''
        node_labels_str = ''
        for _ in range(node_counter):
            graph_indicator_str += '1\n'
            node_labels_str += '1\n'

        with open(f'{raw_path}/{dataset_name}_graph_indicator.txt', 'w') as gi:
            gi.write(graph_indicator_str)

        with open(f'{raw_path}/{dataset_name}_node_labels.txt', 'w') as nl:
            nl.write(node_labels_str)

        with open(f'{raw_path}/{dataset_name}_graph_labels.txt', 'w') as gl:
            gl.write("1\n1\n")

if __name__ == '__main__':
    main()