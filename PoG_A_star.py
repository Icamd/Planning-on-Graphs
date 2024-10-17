from tqdm import tqdm
import argparse
from utils import *
import random
from client import *
from freebase_func import *
import networkx as nx
import datasets
import os
import heapq
from evaluate_results import eval_result
os.environ.pop("http_proxy", None)
os.environ.pop("all_proxy", None)
os.environ.pop("https_proxy", None)
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://api.openai.com/v1",
)

def build_graph(graph: list) -> nx.Graph:
    G = nx.Graph()
    for triplet in graph:
        h, r, t = triplet
        G.add_edge(h, t, relation=r.strip())
    return G

def dfs(graph, current_node, depth, path, all_paths, visited_edges):
    if len(path) == depth + 1:
        all_paths.append(path.copy())
        return

    for neighbor in graph.neighbors(current_node):
        edge = (current_node, neighbor)
        if edge not in visited_edges and (neighbor, current_node) not in visited_edges and current_node != neighbor:
            visited_edges.add(edge)
            path.append(neighbor)
            dfs(graph, neighbor, depth, path, all_paths, visited_edges)
            path.pop()
            visited_edges.remove(edge)

def find_paths_of_depth(graph, start_node, depth):
    all_paths = []
    path = [start_node]
    visited_edges = set()
    dfs(graph, start_node, depth, path, all_paths, visited_edges)
    return all_paths

def merge_rule_result(qa_dataset, rule_dataset, n_proc=1, filter_empty=False):
    question_to_rule = dict()
    for data in rule_dataset:
        qid = data["id"]
        predicted_paths = data["prediction"]
        ground_paths = data["ground_paths"]
        question_to_rule[qid] = {
            "predicted_paths": predicted_paths,
            "ground_paths": ground_paths,
        }

    def find_rule(sample):
        qid = sample["id"]
        sample["predicted_paths"] = []
        sample["ground_paths"] = []
        sample["predicted_paths"] = question_to_rule[qid]["predicted_paths"]
        sample["ground_paths"] = question_to_rule[qid]["ground_paths"]
        return sample  # TODO: ignore the sample with zero paths.

    qa_dataset = qa_dataset.map(find_rule, num_proc=n_proc)
    if filter_empty:
        qa_dataset = qa_dataset.filter(
            lambda x: len(x["ground_paths"]) > 0, num_proc=n_proc
        )
    return qa_dataset

def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def a_star_3(graph, start, len_pred, h_value_list, relation_mapping):
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    relations_mapping = {node: [] for node in graph}
    priorities = {node: float('infinity') for node in graph}
    priority_queue = [(0, 0, start, 0, [], [])]
    came_from = {start: None}

    visited = set()

    while priority_queue:
        current_priority, current_total_distance, current_node, edge_count, current_route, node_path = heapq.heappop(priority_queue)

        if edge_count >= len_pred:
            continue
        if current_node in visited:
            continue
        visited.add(current_node)

        for neighbor, weight in graph[current_node].items():
            new_total_distance = current_total_distance + weight
            new_edge_count = edge_count + 1
            new_average_distance = new_total_distance / new_edge_count

            if new_average_distance < distances[neighbor] and new_edge_count <= len_pred:
                distances[neighbor] = new_average_distance
                new_current_route = current_route.copy()
                new_current_route.append(relation_mapping[current_node][neighbor]['relation'])
                current_node_path_head = node_path.copy()
                current_node_path_head.append(current_node)
                current_node_path_tail = current_node_path_head.copy()
                current_node_path_tail.append(neighbor)

                priority = new_average_distance + h_value_list[' -> '.join(current_node_path_tail)]
                heapq.heappush(priority_queue, (priority, new_total_distance, neighbor, new_edge_count, new_current_route, current_node_path_head))
                came_from[neighbor] = current_node
                came_from[neighbor] = current_node
                relations_mapping[neighbor] = new_current_route
                priorities[neighbor] = priority

    return distances, relations_mapping, priorities

def select_unique_paths(data_dict, top_n):
    sorted_items = sorted(data_dict.items(), key=lambda item: item[1]['value'])

    unique_paths = []
    seen_paths = set()

    for key, sub_dict in sorted_items:
        path = tuple(sub_dict['path'])
        if path not in seen_paths:
            unique_paths.append((path, sub_dict['value']))
            seen_paths.add(path)
        if len(unique_paths) >= top_n:
            break

    return unique_paths


def top_three_indices(lst, width):
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)

    return sorted_indices[:width]

def find_path(backtrace_dict, start_tail_entity):
    path = []
    current_entity = start_tail_entity

    while current_entity in backtrace_dict:
        head_entity, relation = backtrace_dict[current_entity]
        path.append((head_entity, relation, current_entity))
        current_entity = head_entity

    path.reverse()
    return path

def extract_text_segment(text):
    last_period_index = text.rfind('.')

    if last_period_index == -1:
        return ''

    segment_start = text.rfind('.', 0, last_period_index)
    if segment_start == -1:
        return text
    else:
        return text[segment_start + 1:last_period_index + 1].strip()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-4-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    args = parser.parse_args()
    if args.dataset == 'cwq':
        args.rule_path = './data/ground_cwq.jsonl'
        datas, question_string = prepare_dataset(args.dataset)
        datasets = datasets.load_dataset('rmanluo/RoG-cwq', split="test")
        rule_dataset = load_jsonl(args.rule_path)
        datasets = merge_rule_result(datasets, rule_dataset)
        output_file = "./predictions/cwq/predictions.jsonl"
        fout, processed_list = get_output_file(output_file, force=False)
        id_str = 'ID'
    elif args.dataset == 'webqsp':
        args.rule_path = './data/ground_webqsp.jsonl'
        datas, question_string = prepare_dataset(args.dataset)
        datasets = datasets.load_dataset('rmanluo/RoG-webqsp', split="test")
        rule_dataset = load_jsonl(args.rule_path)
        datasets = merge_rule_result(datasets, rule_dataset)
        output_file = "./predictions/webqsp/predictions.jsonl"
        fout, processed_list = get_output_file(output_file, force=False)
        id_str = 'QuestionId'

    for dataset in tqdm(datasets):
        id = dataset['id']
        if id in processed_list:
            continue
        for data in datas:
            if data[id_str]==id:
                break
        question = data[question_string]
        topic_entity = {}
        best_predict_paths = {}
        for key, value in data['topic_entity'].items():
            topic_entity[value] = {0: [value]}
            best_predict_paths[value] = []
        trace_back = {}
        cluster_chain_of_entities = []
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        graph = build_graph(dataset['graph'])

        skip_current_sample = False
        all_node_list = list(graph.nodes())
        for answer in dataset['answer']:
            if answer not in all_node_list:
                skip_current_sample = True
                break
        if skip_current_sample == True:
            continue

        entity_dict = {}
        for entity in topic_entity.keys():
            entity_dict[entity] = {}
            grouped_dict = {}
            for i in range(args.depth + 1):
                grouped_dict[i] = []
            for i in range(args.depth + 1):
                paths = find_paths_of_depth(graph, entity, i)
                for path in paths:
                    if path[i] not in grouped_dict[i]:
                        grouped_dict[i].append(path[i])
            entity_dict[entity]['grouped_dict'] = grouped_dict
            sorted_keys = sorted(grouped_dict.items(), key=lambda x: x[0], reverse=False)
            grouped_keys = [keys for value, keys in sorted_keys]
            relations_embeddings = []
            entity_lists = []
            relation_lists = []
            for i in range(len(grouped_keys) - 1):
                k_hop_relation_list = []
                entity_list = []
                for u in grouped_keys[i]:
                    for v in grouped_keys[i + 1]:
                        if u != v and graph.has_edge(u, v) and graph[u][v]['relation'] not in k_hop_relation_list:
                            k_hop_relation_list.append(graph[u][v]['relation'])
                            entity_list.append([u, v])
                relations_embeddings.append(sentence_emb(k_hop_relation_list))
                entity_lists.append(entity_list)
                relation_lists.append(k_hop_relation_list)
            entity_dict[entity]['relations_embeddings'] = relations_embeddings
            entity_dict[entity]['entity_lists'] = entity_lists
            entity_dict[entity]['relation_lists'] = relation_lists

        results = ''
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:

                if_score_match = len(current_entity_relations_list) < 3
                if entity!="[FINISH_ID]":
                    if i < len(pre_heads):
                        retrieve_relations_with_scores, head_relations, total_relations = relation_search_prune_dijkstra(entity, topic_entity[entity][depth-1], pre_relations, pre_heads[i], question, args, graph, if_score_match)# best entity triplet, entitiy_id
                    else:
                        break

                    if retrieve_relations_with_scores == []:
                        break
                    max_score = -1
                    max_score_dict = None
                    for retrieve_relation in retrieve_relations_with_scores:
                        if retrieve_relation['score'] > max_score:
                            max_score = retrieve_relation['score']
                            max_score_dict = retrieve_relation
                    best_predict_paths[entity].append(max_score_dict['relation'])
                    cosine_scores = util.pytorch_cos_sim(sentence_emb([max_score_dict['relation']]), entity_dict[entity]['relations_embeddings'][depth-1])
                    cost = (1 - cosine_scores[0]).tolist()
                    correspond_relations = entity_dict[entity]['relation_lists'][depth-1]
                    correspond_cost = []
                    for total_relation in total_relations:
                        if total_relation in correspond_relations:
                            index = correspond_relations.index(total_relation)
                            correspond_cost.append(cost[index])
                        else:
                            correspond_cost.append(1.0)
                    indexs = top_three_indices(correspond_cost, args.width)

                    while len(retrieve_relations_with_scores) < 3:
                        retrieve_relations_with_scores.append(retrieve_relations_with_scores[0])

                    j = 0
                    for index in indexs:
                        for head_entity in head_relations:
                            if total_relations[index] in head_relations[head_entity]:
                                retrieve_relations_with_scores[j]['entity'] = head_entity
                                retrieve_relations_with_scores[j]['relation'] = total_relations[index]
                                j = j + 1
                                break

                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:

                if len(find_path(trace_back, entity['entity'])) == 0:
                    temporarily_key = entity['entity']
                else:
                    temporarily_key = find_path(trace_back, entity['entity'])[0][0]

                if entity['head']:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], graph, entity_dict[temporarily_key]['grouped_dict'][depth], True)
                else:
                    entity_candidates_id = entity_search(entity['entity'], entity['relation'], graph, entity_dict[entity['entity']][depth], False)

                if args.prune_tools == "llm":
                    if len(entity_candidates_id) >=20:
                        entity_candidates_id = random.sample(entity_candidates_id, args.num_retain_entity)

                if len(entity_candidates_id) ==0:
                    continue

                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity['score'], entity['relation'], args)

                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head)

            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break

            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                for a, b, c in chain_of_entities[0]:
                    trace_back[c] = [a, b]

            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, file_name=args.dataset)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args)
                        flag_printed = True
                    else:
                        if depth != args.depth:
                            for key in topic_entity:
                                topic_entity[key][depth] = []
                            for key in entities_id:
                                back_trace_path = find_path(trace_back, key)
                                h = back_trace_path[0][0]
                                topic_entity[h][depth].append(key)

                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args)
                flag_printed = True
                break

        new_result = ''
        if depth > 2:
            entity_dict = {}
            for entity in topic_entity:
                entity_dict[entity] = {}
                grouped_dict = {}
                for i in range(args.depth):
                    grouped_dict[i] = []
                for i in range(args.depth):
                    paths = find_paths_of_depth(graph, entity, i)
                    for path in paths:
                        if path[i] not in grouped_dict[i]:
                            grouped_dict[i].append(path[i])
                entity_dict[entity]['grouped_dict'] = grouped_dict
                sorted_keys = sorted(grouped_dict.items(), key=lambda x: x[0], reverse=False)
                grouped_keys = [keys for value, keys in sorted_keys]
                relations_embeddings = []
                entity_lists = []
                for i in range(len(grouped_keys) - 1):
                    k_hop_relation_list = []
                    entity_list = []
                    for u in grouped_keys[i]:
                        for v in grouped_keys[i + 1]:
                            if u != v and graph.has_edge(u, v):
                                k_hop_relation_list.append(graph[u][v]['relation'])
                                entity_list.append([u, v])
                    relations_embeddings.append(sentence_emb(k_hop_relation_list))
                    entity_lists.append(entity_list)
                entity_dict[entity]['relations_embeddings'] = relations_embeddings
                entity_dict[entity]['entity_lists'] = entity_lists
                entity_dict[entity]['relation_cost'] = []
                entity_dict[entity]['all_entity_pair'] = []

            for entity in topic_entity:
                i = 0
                while len(best_predict_paths[entity]) < len(entity_dict[entity]['relations_embeddings']):
                    best_predict_paths[entity].append(best_predict_paths[entity][0])
                for relations_embeddings_list in entity_dict[entity]['relations_embeddings']:
                    cosine_scores = util.pytorch_cos_sim(sentence_emb([best_predict_paths[entity][i]]), relations_embeddings_list)
                    entity_dict[entity]['relation_cost'].extend((1 - cosine_scores[0]).tolist())
                    entity_dict[entity]['all_entity_pair'].extend(entity_dict[entity]['entity_lists'][i])
                    i = i + 1

            h_value_dicts = {}
            for entity in topic_entity:
                max_len = 2
                path_dict = {}
                for length in range(max_len + 1):
                    paths = find_paths_of_depth(graph, entity, length)
                    for path in paths:
                        key = []
                        for i in range(len(path) - 1):
                            current_element = path[i]
                            next_element = path[i + 1]
                            key.append(graph[current_element][next_element]['relation'])

                        key_str = " -> ".join(key)
                        if key_str in path_dict:
                            path_dict[key_str].append(path)
                        else:
                            path_dict[key_str] = [path]

                key_list = list(path_dict.keys())
                embeddings = sentence_emb(key_list)

                h_value_list = {}
                cosine_scores = util.pytorch_cos_sim(sentence_emb([" -> ".join(best_predict_paths[entity])]), embeddings)
                similarity = cosine_scores[0].tolist()
                key_count = 0
                for key in key_list:
                    for value in path_dict[key]:
                        h_value_list[" -> ".join(value)] = 1 - similarity[key_count]
                    key_count = key_count + 1

                h_value_dicts[entity] = h_value_list

            new_cluster_chain_of_entities = []
            for entity in topic_entity:
                dijkstra_graph = {}
                all_paths = {}
                for i in range(len(entity_dict[entity]['all_entity_pair'])):
                    node1, node2 = entity_dict[entity]['all_entity_pair'][i]
                    cost = entity_dict[entity]['relation_cost'][i]

                    if node1 not in dijkstra_graph:
                        dijkstra_graph[node1] = {}
                    if node2 not in dijkstra_graph:
                        dijkstra_graph[node2] = {}

                    dijkstra_graph[node1][node2] = cost

                depth = 2
                shortest_paths, relations_mapping, priorities = a_star_3(dijkstra_graph, entity, depth, h_value_dicts[entity], graph)
                for key, value in priorities.items():
                    if len(relations_mapping[key]) == depth:
                        if key not in all_paths:
                            if value > 0.001:
                                all_paths[key] = {'value': value, 'path': relations_mapping[key]}
                            else:
                                all_paths[key] = {'value': 0.0, 'path': relations_mapping[key]}
                        else:
                            if value < all_paths[key]['value']:
                                all_paths[key] = {'value': value, 'path': relations_mapping[key]}
                if len(topic_entity) > 1:
                    num_path = 5
                else:
                    num_path = 10
                related_paths = select_unique_paths(all_paths, num_path)
                for related_path in related_paths:
                    for key, value in relations_mapping.items():
                        if value == list(related_path[0]):
                            new_cluster_chain_of_entities.append((entity, related_path[0][-1], key))
                            break
            new_stop, new_result = reasoning(question, [[new_cluster_chain_of_entities]], args)
            if new_stop:
                print("Find answer with A_star in %s " % id)

        back_up_result = results
        if True:
            use_cot = False
            use_exhaustivity = False
            if '{Yes}' not in results and '{Yes}' not in new_result:
                results = generate_without_explored_paths(question, args)
                save_2_jsonl(question, results, [], file_name=args.dataset)
                use_cot = True
                extract_answers = []

                for answer in all_node_list:
                    if answer.lower() in results.lower():
                        extract_answers.append(answer)
                use_exhaustivity = True

                extract_answers = list(set(extract_answers))

            elif '{Yes}' not in results and '{Yes}' in new_result:
                final_entities = [[x[2] for x in sublist] for sublist in [[new_cluster_chain_of_entities]][-1]][0]
                extract_answers = []
                results = new_result
                match = extract_text_segment(new_result)

                for answer in final_entities:
                    if answer.lower() in match.lower():
                        extract_answers.append(answer)
                if len(extract_answers) == 0:
                    use_exhaustivity = True
            elif '{Yes}' in results and '{Yes}' in new_result:
                final_entities = [[x[2] for x in sublist] for sublist in cluster_chain_of_entities[-1]][0]
                extract_answers = []
                match = extract_text_segment(results)

                for answer in final_entities:
                    if answer.lower() in match.lower():
                        extract_answers.append(answer)
                final_entities = [[x[2] for x in sublist] for sublist in [[new_cluster_chain_of_entities]][-1]][0]
                extract_answers_new = []
                match = extract_text_segment(new_result)

                for answer in final_entities:
                    if answer.lower() in match.lower():
                        extract_answers_new.append(answer)

                extract_answers.extend(extract_answers_new)

                if len(extract_answers) == 0:
                    use_exhaustivity = True
                results = new_result
            elif '{Yes}' in results and '{Yes}' not in new_result:
                final_entities = [[x[2] for x in sublist] for sublist in cluster_chain_of_entities[-1]][0]
                extract_answers = []
                match = extract_text_segment(results)

                for answer in final_entities:
                    if answer.lower() in match.lower():
                        extract_answers.append(answer)
                if len(extract_answers) == 0:
                    use_exhaustivity = True


            extract_answers = list(set(extract_answers))

            if use_exhaustivity == True:
                for node in graph.nodes():
                    if node.lower() in results.lower():
                        extract_answers.append(node)
                print('start openai')
                modified_text = results
                if modified_text.startswith("{Yes}."):
                    modified_text = modified_text[7:].strip()
                response = client.chat.completions.create(model=args.LLM_type, messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Please select all phrases or words from the reference that are related to the question (for example: names, articles, proper nouns, positions, dates, addresses, etc), and separate the extracted phrases with '/'. \nQuestion: "
                                                + question + "\n" + "Reference: " + modified_text + "\nA:"}],
                                                          max_tokens=200, temperature=0.1, stream=False)
                relative_parts = response.choices[0].message.content.split('/')
                print('end openai')
                cosine_scores = util.pytorch_cos_sim(sentence_emb(relative_parts), sentence_emb(all_node_list))
                retrieved_entities = []
                for i in range(len(relative_parts)):
                    score = cosine_scores[i].tolist()
                    retrieved_entities.append(all_node_list[score.index(max(score))])
                extract_answers.extend(retrieved_entities)
                extract_answers = list(set(extract_answers))

                print('start openai')
                response = client.chat.completions.create(model=args.LLM_type, messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Please select all options from the provided answer list that could potentially serve as answers to the given question, and separate them with '/'. \nQuestion: "
                                                + question + "\n" + "Answer list: " + '/'.join(extract_answers) + "\nA:"}],
                                                          max_tokens=200, temperature=0.1, stream=False)
                relative_parts = response.choices[0].message.content.split('/')
                print('end openai')
                extract_answers = []
                extract_answers = relative_parts.copy()

            extract_answers = list(set(extract_answers))
            for key in topic_entity:
                if key in extract_answers:
                    extract_answers.remove(key)

            answer = '\n'.join(extract_answers)

        format_result = {
            "id": id,
            "question": question,
            "prediction": answer,
            "ground_truth": dataset["answer"],
            "results": results,
            "reasoning_chains": cluster_chain_of_entities
        }
        fout.write(json.dumps(format_result) + "\n")
        fout.flush()

    fout.close()
    eval_result(output_file)