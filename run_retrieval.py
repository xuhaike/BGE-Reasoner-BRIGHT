import numpy as np
import yaml
import diskannpy
import time
import os
import csv
import struct

import pickle
import pandas as pd
import heapq
import bisect
import pytrec_eval

import math

from beir.datasets.data_loader import GenericDataLoader

import logging
import pathlib, os
import sys

from tqdm import tqdm
import random

import json

from datetime import datetime

import copy
from copy import deepcopy
import multiprocessing
from multiprocessing import Pool
import sys
import re

import shutil

import argparse

from pathlib import Path

from openai import OpenAI
from google import genai
from google.genai import types

import matplotlib.pyplot as plt

from datasets import load_dataset

import faiss

import signal

BrightShortInstructions = {
    # StackExchange
    "biology": "Given a Biology post, retrieve relevant passages that help answer the post.",
    "earth_science": "Given an Earth Science post, retrieve relevant passages that help answer the post.",
    "economics": "Given an Economics post, retrieve relevant passages that help answer the post.",
    "psychology": "Given a Psychology post, retrieve relevant passages that help answer the post.",
    "robotics": "Given a Robotics post, retrieve relevant passages that help answer the post.",
    "stackoverflow": "Given a Stack Overflow post, retrieve relevant passages that help answer the post.",
    "sustainable_living": "Given a Sustainable Living post, retrieve relevant passages that help answer the post.",
    # Coding
    "leetcode": "Given a Coding problem, retrieve relevant examples that help answer the problem.",
    "pony": "Given a Pony question, retrieve relevant passages that help answer the question.",
    # Theorem-based
    "aops": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_questions": "Given a Math problem, retrieve relevant examples that help answer the problem.",
    "theoremqa_theorems": "Given a Math problem, retrieve relevant theorems that help answer the problem.",
}

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")

signal.signal(signal.SIGALRM, timeout_handler)

llm_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
api_type = None  # Track which API we're using

if os.environ.get("GEMINI_API_KEY"):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    api_type = "gemini"
    print(f"✓ Using Gemini API: {llm_model_name}")

elif os.environ.get("OPENROUTER_API_KEY"):
    client = OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    # llm_model_name = "google/gemini-2.0-flash-exp:free"

    # llm_model_name =  "google/gemini-2.0-flash-001"
    llm_model_name = "google/gemini-2.5-flash"
    api_type = "openai_compatible"
    print(f"✓ Using OpenRouter API: {llm_model_name}")

elif os.environ.get("OPENAI_API_KEY"):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    llm_model_name = "gpt-4o-mini"
    # llm_model_name = "gpt-5-mini"
    api_type = "openai_compatible"
    print(f"✓ Using OpenAI API: {llm_model_name}")

else:
    raise ValueError("No API key found! Please set GEMINI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_name", type=str, default="BEIR", help="which benchmark data format to use")
    parser.add_argument("--dataset_name", type=str, default=None, help="which data set to use")
    parser.add_argument("--data_type", type=str, default="fp32", help="which data set to use")
    parser.add_argument("--algo_name", type=str, default="diskann", help="which algorithm to use")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the cheap biencoder name")
    parser.add_argument("--expensive_model_name", type=str, default=None, help="Name of the expensive biencoder name")
    parser.add_argument("--embedding_path", type=str, default="./embedding_data", help="Directory to load embeddings")
    parser.add_argument("--graph_degree", type=int, default=-1, help="what max degree to use when building the index")
    parser.add_argument("--time_tag", type=str, default="", help="time tag")
    return parser.parse_args()

args = parse_arguments()

dataset_name=args.dataset_name

benchmark_name=args.benchmark_name

data_type=args.data_type

graph_degree = args.graph_degree
algo_name=args.algo_name

embedding_path=args.embedding_path

time_tag=args.time_tag

print(f"Dataset: {dataset_name}")

split="test"

if benchmark_name=="BRIGHT":
    data = load_dataset('xlangai/BRIGHT', 'examples')[dataset_name]
    documents = load_dataset('xlangai/BRIGHT', 'documents')[dataset_name]

    queries={}
    qrels={}
    for query in data:
        queries[query["id"]]={"text":query["query"]}
        answer={}
        for entry in query["gold_ids"]:
            answer[entry]=1
        qrels[query["id"]]=answer

    corpus={}
    for doc in documents:
        corpus[doc["id"]]={"title":"","text":doc["content"]}
else:
    raise ValueError(f"Unsupported benchmark: {benchmark_name}. Only BRIGHT is supported.")

for qid in queries:
    assert("text" in queries[qid])
for pid in corpus:
    assert("text" in corpus[pid] and "title" in corpus[pid]) 

if graph_degree==-1:
    if len(corpus)>1000000:
        graph_degree=32
    elif len(corpus)>100000:
        graph_degree=16
    elif len(corpus)>10000:
        graph_degree=12
    else:
        graph_degree=8

print("corpus len", len(corpus))
print("query len", len(queries))
print("qrel len", len(qrels))

log_path=f"./experiment_log/{benchmark_name}_{args.model_name}_{args.expensive_model_name}_log/"
csv_path=f"./results/{benchmark_name}_{args.model_name}_{args.expensive_model_name}_csv/"

for directory in [log_path,csv_path]:
    if not os.path.exists(directory):
        os.makedirs(directory)

model_name=args.model_name
expensive_model_name=args.expensive_model_name

metric="expensive"

output_path=os.path.join(log_path,f"{dataset_name}_{model_name}_{expensive_model_name}_{split}_{time_tag}")
output_file=open(output_path,"w")

csv_name=f"{dataset_name}_{model_name}_{expensive_model_name}_{split}_deg{graph_degree}_{time_tag}.csv"

csv_file=open(os.path.join(csv_path,csv_name), 'a', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["algo_name","opt_recall","ndcg_gt","# of bi-encoder evals","# of reranker evals","# of tokens","# of api_calls"])

print(dataset_name)

query=[]
query_id={}
query_name=[]
for qid, value in queries.items():
    query_name.append(qid)
    query_id[qid]=len(query)
    query.append(value["text"])
print("read query complete")

query_embedding_name=os.path.join(embedding_path,os.path.join(f"./{benchmark_name}/{dataset_name}_{split}_{model_name}_query_embeddings.npy"))

print(query_embedding_name)
if os.path.exists(query_embedding_name):
    query_embeddings = np.load(query_embedding_name).astype("float32")
    print("load from file")
else:
    print("no bi-encoder embedding provided")
    assert(False)
print("query embeddings complete", query_embeddings.shape)

def clean_format(x):
    ret=x.replace("[","")
    ret=ret.replace("]","")
    return ret

passage=[]
passage_name=[]
passage_id={}

for pid, value in corpus.items():
    passage_name.append(pid)
    passage_id[pid]=len(passage)
    passage.append(clean_format(value["title"]+" "+value["text"]))

passage_embedding_name=os.path.join(embedding_path,os.path.join(f"./{benchmark_name}/{dataset_name}_{model_name}_passage_embeddings.npy"))

if os.path.exists(passage_embedding_name):
    passage_embeddings=np.load(passage_embedding_name).astype("float32")
    print("load from file")
    print("passage embeddings complete", passage_embeddings.shape)
else:
    print("no bi-encoder embedding provided")
    assert(False)

groundtruth={}
for qid, res in qrels.items():
    groundtruth[qid]=[]
    for pid, value in res.items():
        if value!=0 and pid in passage_id:
            groundtruth[qid].append((value,passage_id[pid]))
    groundtruth[qid]=[x[1] for x in sorted(groundtruth[qid],reverse=True)]

def build_index(ann_algo_name):
    global graph,start,max_deg,num_nodes,in_vis,in_Q,col

    if not os.path.exists(f"./indices/"):
        os.makedirs("./indices")
    directory = "./indices"

    k = 10
    if ann_algo_name=="diskann":

        def get_index_query_parameters(config_path):
            configurations = []
            with open(config_path, "r") as yaml_file:
                config = yaml.safe_load(yaml_file)["float"]["angular"]
            for algo_type in config:
                if algo_type["name"] != "vamana(diskann)":  # Avoids vamana-pq(diskann)
                    continue
                configurations += algo_type["run_groups"].values()
            return configurations

        config_yml_path = "vamana-config.yaml"
        configurations = get_index_query_parameters(config_yml_path)
        configuration=configurations[-1]

        args = configuration["args"][0]
        alpha = args["alpha"]
        complexity=args["l_build"]
        prefix = f"{dataset_name}-{model_name}-{alpha}-{complexity}-{graph_degree}"

        print(prefix)

        if not os.path.exists(directory + "/" + prefix):
            diskannpy.build_memory_index(
                passage_embeddings,
                alpha=alpha,
                complexity=complexity,
                graph_degree=graph_degree,
                distance_metric="cosine",
                index_directory=directory,
                num_threads=0,
                use_pq_build=False,
                use_opq=False,
                index_prefix=prefix,
            )

        index_path=f"./indices/{prefix}"

        with open(index_path, 'rb') as index_path:
            expected_file_size = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]
            max_deg = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
            start = struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
            file_frozen_pts = struct.unpack('Q', index_path.read(struct.calcsize('Q')))[0]

            bytes_read=struct.calcsize('Q')*2+struct.calcsize('I')*2
            i=0
            while bytes_read!=expected_file_size:
                deg=struct.unpack('I', index_path.read(struct.calcsize('I')))[0]
                graph.append(list(struct.unpack(f'{deg}I', index_path.read(deg * 4))))
                bytes_read+=(deg+1)*struct.calcsize('I')
                i+=1

            num_nodes=len(graph)
        in_vis=[0]*num_nodes
        in_Q=[0]*num_nodes
        col=0
    elif ann_algo_name=="knn":

        prefix = f"{dataset_name}-{model_name}-knn-{graph_degree}"

        if os.path.exists(directory + "/" + prefix + ".npy") is False:
            norms = np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
            normalized_passage_embeddings = passage_embeddings / norms

            dim = normalized_passage_embeddings.shape[1]  # Dimensionality of your vectors
            M = 64  # Number of neighbors in the graph
            ef_construction = 200  # Construction parameter
            index = faiss.IndexHNSWFlat(dim, M)
            index.hnsw.efConstruction = ef_construction
            index.add(normalized_passage_embeddings)
            
            ef_search = 1000  # A higher value leads to more accurate but slower searches
            index.hnsw.efSearch = ef_search

            k=graph_degree+1
            D, I = index.search(normalized_passage_embeddings, k)

            np.save(os.path.join(directory + "/" + prefix + ".npy"), I[:,1:])
        index_path=f"./indices/{prefix}.npy"
        graph=np.load(index_path).tolist()
        num_nodes=len(graph)
        max_deg=graph_degree

        in_vis=[0]*num_nodes
        in_Q=[0]*num_nodes
        col=0

    print(f"build and read {ann_algo_name} index complete")

search_results={}

comp_count={}

def cos_sim(A,B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    similarity = dot_product / norm_A / norm_B
    return similarity

def dist(qid,pid,metric="cos",comp_count_factor=1):
    global comp_count

    if metric=="cos":
        score=cos_sim(query_embeddings[qid],passage_embeddings[pid])
        comp_count["biencoder"]+=comp_count_factor
        return 1-score
    elif metric=="l2":
        l2_dist=np.linalg.norm(query_embeddings[qid]-passage_embeddings[pid],ord=2)
        comp_count["biencoder"]+=comp_count_factor
        return l2_dist

def dist_all(qid,pids,metric="cos",comp_count_factor=1):

    if pids==[]:
        return []
 
    scores=[]
    for pid in pids:
        scores.append(dist(qid,pid,metric=metric,comp_count_factor=comp_count_factor))
    return scores

def dist_rank(qid,pids):
    global api_type
    rank_list=[]
    output_validity=True

    # sys_instruct=f"You are an intelligent assistant that can rank answers based on their relevancy to the query. I will provide you with {len(pids)} passages, each indicated by number identifier []. \nRank the answers based on their relevance to query: {query[qid]}."

    sys_instruct=f"You are an expert in {dataset_name}. {BrightShortInstructions[dataset_name]} I will provide you with a query and {len(pids)} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query."

    start_time=time.time()
    call_success=True
    output=None

    try:
        signal.alarm(30)
        start_time=time.time()

        passages_text = "\n\n".join([f"[{i+1}] {passage[pid]}" for i,pid in enumerate(pids)])
        user_message = f"Query: {query[qid]}.\n Passages: {passages_text}\n\n \nRank the {len(pids)} passages above based on their relevance to the query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be like [1] > [2] ... > [{len(pids)}]. Only response the ranking results, do not say any word or explain."

        char_count = len(sys_instruct) + len(user_message)
        total_tokens = char_count // 4

        if api_type == "gemini":
            messages = [sys_instruct, f"Query: {query[qid]}"]
            for i,pid in enumerate(pids):
                messages.append(f"[{i+1}] {passage[pid]}")
            messages.append(f"Rank the {len(pids)} passages above based on their relevance to the query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be like [1] > [2] ... > [{len(pids)}]. Please read these passages carefully, one by one, to determine their order. Only respond with the ranking results; do not include any additional text or explanation.")

            response = client.models.generate_content(
                model=llm_model_name,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=window_size*10,
                    system_instruction=sys_instruct),
                contents=messages
            )
            output = response.text

        elif api_type == "openai_compatible":
            response = client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": sys_instruct},
                    {"role": "user", "content": user_message}
                ],
                temperature=0,
                max_tokens=window_size*10
            )
            output = response.choices[0].message.content

        end_time=time.time()
        signal.alarm(0)

        comp_count["tokens"]+=total_tokens
        comp_count["api_calls"]+=1

    except Exception as e:
        signal.alarm(0)
        print(f"LLM API call failed: {e}")
        sleep_time = random.uniform(0, 4)
        time.sleep(sleep_time)
        call_success=False

    end_time=time.time()

    rank_list=[]
    output_validity=True
    if call_success==True and type(output)!=type(None):
        for i in range(len(pids)):
            if f"[{i+1}]" in output:
                rank_list.append((output.find(f"[{i+1}]"),i))
            else:
                output_validity=False
                rank_list.append((len(output)+dist(qid,pids[i],metric="cos",comp_count_factor=0),i))
    else:
        output_validity=False

    if output_validity is False:
        comp_count["invalid"]+=1
        return pids
    else:
        rank_list=sorted(rank_list)
        ret=[]
        for i in range(len(pids)):
            ret.append(pids[rank_list[i][1]])
        return ret

graph=[]
start=0
max_deg=0
num_nodes=0

in_Q=[]
in_vis=[]
col=0

def greedy_search(qid,k_neighbors,search_L,start,metric="cos"):
    global graph,in_Q,in_vis,col

    col+=1

    vis=[]
    if isinstance(start,list):
        cur_dist=[]
        for pid in start:
            cur_dist.append(dist(qid,pid,metric=metric))
        Q=[]
        for i in range(len(start)):
            Q.append((cur_dist[i],start[i]))
            in_Q[start[i]]=col
        heapq.heapify(Q)
    else:
        Q=[(dist(qid,start,metric),start)]
        in_Q[start]=col

    p=Q[0][1]
    while p!=-1:
        if in_vis[p]!=col:
            bisect.insort(vis,(Q[0][0],p))
            in_vis[p]=col
        heapq.heappop(Q)

        V=[x for x in graph[p] if in_Q[x]!=col]
        cur_dist=dist_all(qid,V,metric)

        for i in range(len(V)):
            x=V[i]
            if in_Q[x]!=col:
                heapq.heappush(Q,(cur_dist[i],x))
                in_Q[x]=col

        p=-1
        if len(Q)>0:
            if len(vis)<search_L or Q[0][0]<=vis[search_L-1][0]:
                p=Q[0][1]

    for i in range(min(len(Q),k_neighbors)):
        if in_vis[Q[0][1]]!=col:
            bisect.insort(vis,(Q[0][0],Q[0][1]))
        heapq.heappop(Q)

    neighbors=[]
    distances=[]
    for i in range(min(k_neighbors,len(vis))):
        neighbors.append(vis[i][1])
        distances.append(vis[i][0])

    len_Q=len(Q)
    len_V=len(V)

    return neighbors

def insert(qid,Q,x):
    global comp_count,in_Q,col
    
    if len(x)==0:
        return

    comp_count["expensive"]+=len(x)
    for pid in x:
        Q.append(pid)
    for i in range(1):
        j=len(Q)
        while(j>0):
            new_pid=False
            for pid in Q[max(0,j-window_size):j]:
                if in_Q[pid]!=col:
                    new_pid=True
            if new_pid is True:
                Q[max(0,j-window_size):j]=dist_rank(qid,Q[max(0,j-window_size):j])
            else:
                break
            j=max(0,j-window_size//2)

def greedy_search_cmp_quota(qid,k_neighbors,query_quota,start,metric="llm"):
    global graph,in_Q,in_vis,col

    global comp_count

    if query_quota==100:
        queue_size=20
    elif query_quota==200:
        queue_size=20
    elif query_quota==300:
        queue_size=30
    elif query_quota==500:
        queue_size=50
    else:
        queue_size=50

    current_dist_count=0

    neighbors_seen=[]

    col+=1

    if isinstance(start,list):
        if len(start)>query_quota:
            start=start[:query_quota]
        Q=[]
        insert(qid,Q,start)
        neighbors_seen+=start
        for x in start:
            in_Q[x]=col
        Q=Q[:queue_size]
        current_dist_count+=len(start)
    else:
        Q=[start]
        in_Q[start]=col
        current_dist_count=1

    id=0
    while id<len(Q) and current_dist_count<query_quota:
        p=Q[id]
        in_vis[p]=col
        V=[x for x in graph[p] if in_Q[x]!=col]
        if current_dist_count+len(V)>query_quota:
            V=V[:query_quota-current_dist_count]

        insert(qid,Q,V)

        neighbors_seen+=V

        for pid in V:
            in_Q[pid]=col
        Q=Q[:queue_size] 
        current_dist_count+=len(V)  

        id=0
        while id<len(Q) and in_vis[Q[id]]==col:
            id+=1

    neighbors=[]
    for x in Q[:k_neighbors]:
        neighbors.append(x)

    return neighbors,neighbors_seen

def SlideGAR(qid, pids, k_neighbors,query_quota,metric="llm"):
    start_time=time.time()

    L=[]
    for i in range(window_size):
        L.append(pids[i])
    R1=[]
    i=window_size
    current_dist_count=0
    iteration=0

    while current_dist_count<=query_quota:
        B=dist_rank(qid,L)

        if iteration==0:
            current_dist_count=len(L)
            comp_count["expensive"]+=len(L)
        else:
            current_dist_count+=len(L)-window_size//2
            comp_count["expensive"]+=len(L)-window_size//2

        L1=B[:window_size//2]

        if current_dist_count>=query_quota:
            break

        Frontier=[]
        for x in B:
            for y in graph[x]:
                if (y not in Frontier) and (y not in B) and (y not in R1):
                    Frontier.append(y)
                    if len(Frontier)>=window_size//2:
                        break
        R1=B[window_size//2:]+R1
        if iteration%2==0:
            L=L1+Frontier[:window_size//2]
        else:
            L=L1
            while i < len(pids):
                if (pids[i] not in R1) and (pids[i] not in L):
                    L.append(pids[i])
                i+=1
                if len(L)>=window_size:
                    break

        if current_dist_count+len(L)-len(L1)>query_quota:
            L=L[:len(L1)+query_quota-current_dist_count]
        iteration+=1
    R1=L1+R1

    neighbors_seen=R1
    neighbors=R1[:k_neighbors]
    distances=[i/k_neighbors for i in range(k_neighbors)]

    end_time=time.time()
    return neighbors,distances,neighbors_seen

def rerank(qid, pids, k_neighbors,metric="cos"):
    global comp_count,in_Q,col

    start_time=time.time()

    col+=1

    if metric=="llm":
        Q=[]
        insert(qid,Q,pids)
        Q=Q[:k_neighbors]
        neighbors=Q[:k_neighbors]
        distances=[i/k_neighbors for i in range(k_neighbors)]
        return neighbors[:k_neighbors], distances[:k_neighbors]
    else:
        print("unsupported metric")

    end_time=time.time()

def process_query(qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota):
    global comp_count,cos_threshold

    comp_count={"biencoder":0,"expensive":0,"invalid":0,"processed_query":0,"time":0,"tokens":0,"api_calls":0}

    comp_count["processed_query"]+=1

    assert(query_complexity==0 or query_quota==0)
    assert(second_query_complexity==0 or second_query_quota==0)

    start_time=time.time()

    if (qid,query_complexity,"complexity") in search_results:
        retrieval_neighbors=search_results[(qid,query_complexity,"complexity")]
    else:
        retrieval_neighbors=greedy_search(query_id[qid],k_neighbors=query_complexity,search_L=query_complexity,start=start,metric="cos")

    if retrieval_algo=="bi(llm-baseline)":
        second_L=max(k_neighbors,second_query_quota)

        expensive_neighbors,expensive_distances=rerank(query_id[qid],copy.deepcopy(retrieval_neighbors[:second_L]),k_neighbors=k_neighbors,metric="llm")

        neighbors=expensive_neighbors[:k_neighbors]
        distances=expensive_distances[:k_neighbors]

        neighbors_seen=retrieval_neighbors[:second_query_quota]
    elif retrieval_algo=="bi(llm-SlideGAR)":

        second_L=max(k_neighbors,second_query_quota)

        expensive_neighbors,expensive_distances,neighbors_seen=SlideGAR(query_id[qid],copy.deepcopy(retrieval_neighbors[:second_L]),k_neighbors=k_neighbors,query_quota=second_query_quota,metric="llm")

        neighbors=expensive_neighbors[:k_neighbors]
        distances=expensive_distances[:k_neighbors]
    elif retrieval_algo=="bi(llm-ours)":
        in_Q=[]
        in_vis=[]

        second_L=int(second_query_quota//5)

        expensive_neighbors,neighbors_seen=greedy_search_cmp_quota(query_id[qid],k_neighbors=k_neighbors,query_quota=second_query_quota,start=start if second_L==0 else retrieval_neighbors[:second_L],metric="llm")

        neighbors=expensive_neighbors[:k_neighbors]
        distances=[i/k_neighbors for i in range(k_neighbors)]
    else:
        neighbors=retrieval_neighbors[:k_neighbors]
        distances=dist_all(query_id[qid],neighbors,metric="cos",comp_count_factor=0)

        expensive_neighbors=neighbors
        neighbors_seen=neighbors

    end_time=time.time()
    comp_count["time"]=end_time-start_time

    count_gt=0
    count_opt=0
    count_final=0
    for p_name in qrels[qid]:
        if qrels[qid][p_name]>0:
            pid=passage_id[p_name]
            count_gt+=1
            if pid in neighbors_seen:
                count_opt+=1
            if pid in expensive_neighbors:
                count_final+=1

    count_gt=min(count_gt,10)
    count_opt=min(count_opt,10)
    count_final=min(count_final,10)

    opt_recall=count_opt/count_gt
    final_recall=count_final/count_gt

    ret={qid:{}}
    for i in range(len(neighbors)):
        pid=passage_name[neighbors[i]]
        distance=distances[i]
        ret[qid][pid]=1-float(distance)

    ret[qid]=dict(sorted(ret[qid].items(), key=lambda item: item[1], reverse=True))

    ret["opt_recall"]=opt_recall
    ret["final_recall"]=final_recall

    return {**ret,**comp_count}

def calculate_ndcg(qrels, answers, cutoffs):
    run = {}
    for qid, doc_scores in answers.items():
        run[qid] = {}
        for doc_id, score in doc_scores.items():
            if doc_id!=qid:
                run[qid][doc_id] = score

    metrics_to_evaluate = {f"ndcg_cut.{x}": x for x in cutoffs}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_evaluate)
    results = evaluator.evaluate(run)

    ndcg_averages = []
    for cutoff in cutoffs:
        ndcg_key = f"ndcg_cut_{cutoff}"
        ndcg_values = [res[ndcg_key] for res in results.values() if ndcg_key in res]
        if ndcg_values:
            ndcg_averages.append(sum(ndcg_values) / len(ndcg_values))
        else:
            ndcg_averages.append(0.0)
    
    return ndcg_averages

def retrieval_test(retrieval_algo="our",k_neighbors=10,query_complexity=0,query_quota=0,second_query_complexity=0,second_query_quota=0):
    global metric,comp_count,in_Q,in_vis,cos_threshold,default_best,openai_scores

    time_count=0

    print("query L", query_complexity)
    print("query L", query_complexity,file=output_file)
    print("query quota", query_quota)
    print("query quota", query_quota,file=output_file)
    print("second query complexity",second_query_complexity)
    print("second query complexity",second_query_complexity,file=output_file)
    print("second query quota",second_query_quota)
    print("second query quota",second_query_quota,file=output_file)

    predictions_trec={}
    predictions_trec_cr={}
    comp_count={"biencoder":0,"expensive":0,"invalid":0,"processed_query":0,"time":0,"pids":{},"tokens":0,"api_calls":0}
    omitted=[]
    answer_len=[]
    wrong_stats=[]

    seed_value = 410
    random.seed(seed_value)

    if True:
        num_processes=36
        
        query_args=[]
        for qid in groundtruth:
            query_args.append((qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota))

        # query_args=query_args[:10]

        random.shuffle(query_args)

        results=[]
        i=0
        while i<len(query_args):
            last=min(len(query_args),i+500)
            with Pool(processes=num_processes) as pool:

                def signal_handler(sig, frame):
                    print('Ctrl+C pressed. Terminating processes...')
                    pool.terminate()  # Terminate the pool
                    pool.join()  # Wait for the pool to finish processing
                    sys.exit(0)  # Exit the program

                signal.signal(signal.SIGINT, signal_handler)
                results+=pool.starmap(process_query, query_args[i:last])
            i=last
    else:
        results=[]
        for qid in groundtruth:
            results.append(process_query(qid,retrieval_algo,k_neighbors,query_complexity,query_quota,second_query_complexity,second_query_quota))
            if len(results)>=5:
                break

    opt_recall_list=[]
    final_recall_list=[]

    predictions_trec={}
    for res in results:
        for key,value in res.items():
            if key in ["biencoder","expensive","invalid","processed_query","time","tokens","api_calls"]:
                comp_count[key]+=value
            elif key=="opt_recall":
                opt_recall_list.append(value)
            elif key=="final_recall":
                final_recall_list.append(value)
            else:
                predictions_trec[key]=value

    avg_opt_recall=sum(opt_recall_list)/len(opt_recall_list)
    avg_final_recall=sum(final_recall_list)/len(final_recall_list)

    time_count=comp_count["time"]

    sample_qid = next(iter(predictions_trec))
    if retrieval_algo=="bi" and algo_name=="diskann":
        for qid in predictions_trec:
            distances=[]
            for pid,score in predictions_trec[qid].items():
                distances.append((1-score,passage_id[pid]))
            distances=sorted(distances)
            neighbors=[x[1] for x in distances]
            distances=[x[0] for x in distances]
            if query_complexity!=0:
                search_results[(qid,query_complexity,"complexity")]=neighbors
            else:
                search_results[(qid,query_quota,"quota")]=neighbors

    comp_count["biencoder"]/=comp_count["processed_query"]
    comp_count["expensive"]/=comp_count["processed_query"]
    comp_count["invalid"]/=comp_count["processed_query"]
    comp_count["tokens"]/=comp_count["processed_query"]
    comp_count["api_calls"]/=comp_count["processed_query"]
    ndcg_score=0

    print("avg invalid per query", comp_count["invalid"])
    print("avg invalid per query", comp_count["invalid"],file=output_file)

    test_predictions_trec=deepcopy(predictions_trec)

    ndcg_score_gt = calculate_ndcg(qrels, test_predictions_trec, [1,5,10])

    print("NDCG: ", ndcg_score_gt)

    output_file.flush()

    plot_algo_name={
        "bi":"Retrieve",
        "bi(llm-ours)":"RetrieveGuidedSearch",
        "bi(llm-baseline)":"Retreve-and-Rerank",
        "bi(llm-SlideGAR)":"SlideGAR"
    }

    csv_writer.writerow([plot_algo_name[retrieval_algo],avg_opt_recall,ndcg_score_gt,comp_count["biencoder"],comp_count["expensive"],comp_count["tokens"],comp_count["api_calls"]])
    csv_file.flush()

if algo_name=="RGS":
    build_index("diskann")
    first_query_complexity=5000
    window_size=20
    retrieval_test(retrieval_algo="bi",k_neighbors=first_query_complexity,query_complexity=first_query_complexity)
    for second_query_quota in [100,300,500]:
    # for second_query_quota in [100,300]:
        retrieval_test(retrieval_algo="bi(llm-ours)",query_complexity=first_query_complexity,second_query_quota=second_query_quota)
elif algo_name=="RR":
    build_index("diskann")
    first_query_complexity=5000
    window_size=20
    for second_query_quota in [100,300,500]:
    # for second_query_quota in [100]:
        retrieval_test(retrieval_algo="bi(llm-baseline)",query_complexity=first_query_complexity,second_query_quota=second_query_quota)
elif algo_name=="SlideGAR":
    
    build_index("knn")
    first_query_complexity=5000
    window_size=20
    for second_query_quota in [100,300,500]:
        retrieval_test(retrieval_algo="bi(llm-SlideGAR)",query_complexity=first_query_complexity,second_query_quota=second_query_quota)
