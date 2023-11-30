
import glob
import pickle
import os
import json

def create_dataset_dirs(dataset_name):
    paths = [f'data/{dataset_name}', f'data/{dataset_name}/{dataset_name}',
             f'data/{dataset_name}/{dataset_name}/raw', f'data/{dataset_name}/{dataset_name}/processed']
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

def main():
    datasets_path = "bionic_data"
    datasets = glob.glob(f'{datasets_path}/*.txt')
    A_str = ""
    node_counter = 0
    graph_indicator_str = ''
    node_labels_str = ''
    graph_label_str = ''
    with open(f'{datasets_path}/yeast_IntAct_complex_labels.json') as labels_file:
        labels_obj = json.load(labels_file)
    label_names = set([labels_obj[key][0] for key in labels_obj.keys()])
    labels_name_id_dict = dict()
    for i, name in enumerate(label_names):
        labels_name_id_dict[name] = i + 1
    node_label_dict = dict()
    for key, value in labels_obj.items():
        label_id = labels_name_id_dict[value[0]]
        node_label_dict[key] = label_id
    node_label_counts = [0] * (len(label_names))
    for value in node_label_dict.values():
        node_label_counts[value - 1] += 1
    print(node_label_counts)
    print(sum(node_label_counts))
    to_remove_labels = []
    for i, x in enumerate(node_label_counts):
        if x < 10:
            to_remove_labels.append(i + 1)
    new_node_label_dict = dict()
    for key, value in node_label_dict.items():
        if value not in to_remove_labels:
            new_node_label_dict[key] = value
    node_label_dict = new_node_label_dict
    node_label_counts1 = [0] * (len(label_names))
    for value in node_label_dict.values():
        node_label_counts1[value - 1] += 1
    print(node_label_counts1)
    print(sum(node_label_counts1))
    filtered_label_names = []
    for i, name in enumerate(label_names):
        if (i - 1) in to_remove_labels:
            filtered_label_names.append(name)
    filtered_labels_name_id_dict = dict()
    for i, name in enumerate(filtered_label_names):
        filtered_labels_name_id_dict[name] = i + 1
    filtered_node_label_dict = dict()
    for key, value in labels_obj.items():
        if value[0] in filtered_labels_name_id_dict:
            label_id = filtered_labels_name_id_dict[value[0]]
            filtered_node_label_dict[key] = label_id
    node_label_dict = filtered_node_label_dict
    top_dataset_name = 'BIONIC'
    # Create the download file paths:
    create_dataset_dirs(top_dataset_name)

    raw_path = f'data/{top_dataset_name}/{top_dataset_name}/raw'

    for i, file in enumerate(datasets):
        graph_id = i + 1
        graph_label_str += "1\n"
        dataset_name = file.split('.')[0].split('\\')[1]
        node_dict = dict()

        def get_node_id(node_name, graph_id):
            nonlocal node_counter
            nonlocal graph_indicator_str
            nonlocal node_labels_str
            if node_name not in node_dict:
                node_counter += 1
                if node_name not in node_label_dict:
                    node_labels_str += f'{len(filtered_label_names)}\n'
                else:
                    node_labels_str += f"{node_label_dict[node_name]}\n"
                node_dict[node_name] = node_counter
                graph_indicator_str += f"{graph_id}\n"
            return node_dict[node_name]

        with open(file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            split = line.split(' ')
            if len(split) == 3:
                node_1, node_2, weight = line.split(' ')
            elif len(split) == 2:
                node_1, node_2 = line.split(' ')
                weight = "1.0\n"
            node_1_id = get_node_id(node_1, graph_id)
            node_2_id = get_node_id(node_2, graph_id)
            #new_file_str += f"{node_1_id},{node_2_id},{weight}"
            A_str += f"{node_1_id},{node_2_id}\n"

        with open(f'{raw_path}/node_dict_{dataset_name}_{graph_id}.pickle', 'wb') as handle:
            pickle.dump(node_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{raw_path}/{top_dataset_name}_node_labels.pickle', 'wb') as nl:
        pickle.dump(labels_name_id_dict, nl, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{raw_path}/{top_dataset_name}_A.txt', 'w') as f_A:
        f_A.write(A_str)
        # if need to open the node_dict later use this:
        # with open(f'node_dict_{dataset_name}.pickle', 'rb') as handle:
        #     b = pickle.load(handle)

    with open(f'{raw_path}/{top_dataset_name}_graph_indicator.txt', 'w') as gi:
        gi.write(graph_indicator_str)


    with open(f'{raw_path}/{top_dataset_name}_node_labels.txt', 'w') as nl:
        nl.write(node_labels_str)

    with open(f'{raw_path}/{top_dataset_name}_graph_labels.txt', 'w') as gl:
        gl.write(graph_label_str)


if __name__ == '__main__':
    main()