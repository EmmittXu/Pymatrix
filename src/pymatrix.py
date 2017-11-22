import click
from utils.parsing import parse_options, my_parse_options

import json
import numpy as np
import pandas as pd
import pickle, csv
import sys, operator
from numpy.linalg import norm

@click.group()
def main():
    """
    pymatrix: A command line tool for working with matrices.
    Completed by Guowei Xu, gx2127@columbia.edu, 917-378-2092
    """
    pass


@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
def echo(n, **kwargs):
    """
    Display the passed options N times
    """

    input_type, value = parse_options(**kwargs)

    for _ in range(n):
        click.echo(
            "\nThe given input was of type: %s\n"
            "And the value was: %s\n"
            %(input_type, value)
        )


#A helper function to handle four types of input data
def load_data(input_type, value):
    data=[]
    if(input_type=="json_data"):
        data=np.array(json.loads(value))
    elif(input_type=="csv_file"):
        data=np.array(pd.read_csv(value, header=None))
    elif(input_type=="pickle_file"):
        with open(value, 'r') as f:
            data=np.array(pickle.load(f))
    else:
        #coordinates2array function takes path-to-.coofile as input
        #output a numpy array
        data=coordinates2array(value)
    return data

#Handle coordinate file
#coordinates2array function takes path-to-.coofile as input
#output a numpy array
def coordinates2array(path):
    text=[]
    with open("data/sample.coo", "r") as f:
        text=f.readlines()
    m,n=text[len(text)-1].split()[0], text[len(text)-1].split()[1]
    m,n=int(m), int(n)
    data=np.zeros(shape=(m+1,n+1))
    for line in text:
        data[int(line.split()[0])][int(line.split()[1])]=float(line.split()[2])
    return data

#Calculate Euclidean distance using matrix vectorization
#This substantially increases computation efficiency when matrix dimension is large
def compute_euclidean(data, row_i):
    return np.sqrt(np.sum(np.power((data-data[row_i]), 2), axis=1))

def find_min_distance(row_i, euclidean):
    min_value=sys.maxint
    m=euclidean.shape[0]
    closest_index=0
    for i in range(m):
        if(i==row_i):
            continue
        if(euclidean[i]<min_value):
            min_value=euclidean[i]
            closest_index=i
    if(row_i<closest_index):
        return row_i, closest_index, min_value
    else:
        return closest_index, row_i, min_value

@click.command()
@click.argument('row_i', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL,
              help='print the distance between the pair of rows')
def closest_to(row_i, **kwargs):
    """
    Find the row that is the minimal distance from row_i and
    optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """
    #Pass arguments
    args1, args2= my_parse_options(**kwargs)
    option=args1[1]
    input_type, value=args2[0], args2[1]

    #load text (file) into numeric array
    data=load_data(input_type, value)
    m, n=data.shape

    #Compute euclidean distance between row_i and all the other rows
    euclidean=compute_euclidean(data, row_i)

    #Find the cloesest row and the distance
    output1, output2, distance=find_min_distance(row_i, euclidean)
    if(option==True):
        click.echo(
            "%s %s %s"
            %(output1, output2, distance)
            )
    else:
        click.echo(
            "%s %s"
            %(output1, output2)
            )

@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL,
              help='print the distance between the pair of rows')
def closest(n, **kwargs):
    """
    Find the N distinct pairs of rows that are the smallest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    args1, args2= my_parse_options(**kwargs)
    option=args1[1]
    input_type, value=args2[0], args2[1]

    data=load_data(input_type, value)
    distance=[]
    index=[]
    res=[]

    m=data.shape[0]
    #m=number of rows in matrix, n may be larger than m
    #But the max value for n should be m*(m-1)/2
    assert n<=(m*(m-1)/2), "n should NOT exceed  %s, but you entered %s" %((m*(m-1)/2), n)
    for i in range(m):
        for j in range(i+1, m):
            distance.append(norm(data[i]-data[j]))
            index.append(list([i, j]))
    top_n_index=np.array(distance).argsort(kind='mergesort')[:n]

    for i in top_n_index:
        if(option==True):
            click.echo(
            "%s %s %s" %(index[i][0], index[i][1], distance[i])
                )
        else:
            click.echo(
            "%s %s" %(index[i][0], index[i][1])
            )

@click.command()
@click.argument('n', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')
@click.option('--distance', default=False, type=click.BOOL,
              help='print the distance between the pair of rows')
def furthest(n, **kwargs):
    """
    Find the N distinct pairs of rows that are the furthest distance
    apart and optionally display the distance as well

    Output Format:\n
      i j [d_ij]
    """

    args1, args2= my_parse_options(**kwargs)
    option=args1[1]
    input_type, value=args2[0], args2[1]

    data=load_data(input_type, value)
    distance=[]
    index=[]
    res=[]

    m=data.shape[0]
    #m=number of rows in matrix, n may be larger than m
    #But the max value for n should be m*(m-1)/2
    assert n<=(m*(m-1)/2), "n should NOT exceed  %s, but you entered %s" %((m*(m-1)/2), n)

    for i in range(m):
        for j in range(i+1, m):
            distance.append(norm(data[i]-data[j]))
            index.append(list([i, j]))
    top_n_index=(-np.array(distance)).argsort(kind='mergesort')[:n]

    for i in top_n_index:
        if(option==True):
            click.echo(
            "%s %s %s" %(index[i][0], index[i][1], distance[i])
                )
        else:
            click.echo(
            "%s %s" %(index[i][0], index[i][1])
            )





#########################################################################
#Code for "Centroids" starts here
##########################################################################
#For each vector, compute the distance to each centroid in list centroids
def compute_euclidean2(centroids, vector):
    return abs(np.sum(np.power((centroids-vector), 2), axis=1))

def initialize_centroids(k, data):
    """returns k centroids from the initial points"""

    rand = range(data.shape[0])
    np.random.shuffle(rand)
    centroids = data[rand[:k]]
    return centroids

#Given inputs:data and current centroids,
#output new centroids and new clusters
def update(data, centroids):
    clusters=[[] for _ in range(centroids.shape[0])]
    new_centroids=np.zeros(shape=centroids.shape, dtype=np.float64)
    for vector in data:
        #Euclidean distance between vector and each centroid vector
        euclidean=compute_euclidean2(centroids, vector)
        index=np.argmin(euclidean)
        clusters[index].append(vector)
    for i in range(len(clusters)):
        num_vectors=len(clusters[i])
        new_centroids[i]=np.sum(np.array(clusters[i], dtype=np.float64), axis=0)/num_vectors
    return new_centroids, clusters

#Compute the sum of Euclidean distance between each row vector and its corresponding centroid
def compute_distance(clusters, centroids):
    total=0.0
    for i in range(centroids.shape[0]):
        total+=np.sum(compute_euclidean2(np.array(clusters[i]), centroids[i]), axis=0)
    return total

#This is just one single time of kmeans
def single_kmeans(data, k, max_interation):
    centroids=initialize_centroids(k, data)
    total=sys.maxint
    iteration=0
    any_change=True
    while(any_change==True and iteration<max_interation):
        iteration+=1
        new_centroids, clusters=update(data, centroids)
        if(np.array_equal(new_centroids,centroids)):
            any_change=False
        else:
            centroids=new_centroids
        total=compute_distance(clusters, centroids)
    return total, centroids

#In order to avoid getting stcuk because of one bad centroids initialization
#We perform multiple single_kmeans and take the best result out of it
def my_kmeans(data, k, max_interation=300, num_single_kmeans=10):
    total=sys.maxint
    best_centroids=np.zeros(shape=(k,data.shape[1]))
    for _ in range(num_single_kmeans):
        current_total, current_centroids=single_kmeans(data, k, max_interation)
        if(current_total<total):
            total=current_total
            best_centroids=current_centroids
    return best_centroids.reshape(k,-1), total

@click.command()
@click.argument('n_centroids', type=click.INT)
@click.option('-j', '--json-data', type=click.STRING,
              help='input matrix as a valid json object')
@click.option('-f', '--csv-file', type=click.Path(exists=True),
              help='read matrix from a csv file')
@click.option('-p', '--pickle-file', type=click.Path(exists=True),
              help='read matrix from a pickle file')
@click.option('-s', '--sparse-coo', type=click.Path(exists=True),
              help='read matrix in COO format from a file')

def centroids(n_centroids, **kwargs):
    """
    Cluster the given data set and return the N centroids,
    one for each cluster
    """

    n_init = click.prompt('Please enter the number of times k-means will be run with different centroid initialization default=',default=10, type=int)
    max_interation=click.prompt('Please enter maximum number of iterations of the k-means algorithm for a single run, default=', default=300, type=int)
    input_type, value= parse_options(**kwargs)

    data=load_data(input_type, value)
    assert n_centroids>0 and n_centroids<=data.shape[0], "n_centroids should be between 1 and number of rows=%s" %(data.shape[0])
    best_centroids, best_inertia=my_kmeans(data, n_centroids, max_interation, n_init)

    for centroid in best_centroids:
        click.echo(
        "%s\n"
        %(centroid)
        )
    #print inertia
    click.echo("Best inertia (Sum of L2 distances of samples to their closest cluster center) is %s" %(best_inertia))


main.add_command(echo)
main.add_command(closest_to)
main.add_command(closest)
main.add_command(furthest)
main.add_command(centroids)
