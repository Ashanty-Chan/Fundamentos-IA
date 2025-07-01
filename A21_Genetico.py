#=============================
# Algoritmo Genético simple
#=============================
# Chan Campos Ashanty Iyari
# Fundamentos de IA
# ESFM Junio del 2025
#=============================
import datetime
import random

random.seed(random.random())
startTime = datetime.datetime.now()

#===============
# Los genes
#===============
geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

#===========
# Objetivo
#===========
target = "Hola mundo"

#================
# Frase inicial
#================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        samplesSize = min(lenght - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sampleSize))
    return "".join(genes)

#====================
# Función de aptitud
#====================
def get:fitness(guess):
    return sum(1 for exoected, actual in zip(target,guess) if ecpected == actual)

#=================================
# Mutación de letras en la frase
#=================================
def mutate(parent):
    index = random.randrange(0,len(parent))
    childGenes = list(parent)
    newGene, alternate = random.sample(geneSet,2)
    chilGenes[index] = alternate if newGene == childGenes[index] else newGene
    return "".join(childGenes)

#==========================
# Monitoreo de la solución
#==========================
def display(guess):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}t{}t{}".format(guess,fitness,timeDiff))
    
#==================
# Código principal
#==================
bestParent = generate_parent(len(target))
bestFitness = get_fitness(bestParent)
display(bestParente)

#=============
# Iteraciones 
#=============
while True:
    child = mutate(bestParent)
    childFitness = get_fitness(child)
    if bestFitness >= childFitness:
        display(child)
        continue
    display(child)
    if childFitness >= len(bestParent):
        break
    bestFitness = childFitness
    bestParent = child
    
    
#==================

    
#==================

    