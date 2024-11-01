using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random;

public class GeneticManager : MonoBehaviour
{
    [Header("References")]
    public Ride controller;

    [Header("Controls")]
    public int initPop = 85;
    [Range(0.0f, 1.0f)]
    public float mutRate = 0.055f;

    [Header("Crossover Controls")]
    public int bestAgentSel = 8;
    public int worstAgentSel = 3;
    public int numToCross;


    private List<int> genePool = new List<int>();

    private int naturSel;

    private NNet[] population;

    [Header("Public View")]
    public int currGen;
    public int currGenome = 0;

    private void Start()
    {
        CreatePopulation();
    }

    private void CreatePopulation()
    {
        population = new NNet[initPop];
        FillPopulationWithRandomValues(population, 0);
        ResetToCurrentGenome();
    }

    private void ResetToCurrentGenome()
    {
        controller.ResetWithNetwork(population[currGenome]);
    }

    private void FillPopulationWithRandomValues(NNet[] newPop, int startIndex)
    {
        while (startIndex < initPop)
        {
            newPop[startIndex] = new NNet();
            newPop[startIndex].Initialise(controller.LAYERS, controller.NEURONS);
            startIndex++;
        }
    }


    public void Death(float fitness, NNet network)
    {

        if (currGenome < population.Length - 1)
        {

            population[currGenome].fitness = fitness;
            currGenome++;
            ResetToCurrentGenome();

        }
        else
        {
            RePopulate();
        }

    }

    private void RePopulate()
    {
        genePool.Clear();
        currGen++;
        naturSel = 0;
        SortPopulation();

        NNet[] newPop = PickBestPopulation();

        Crossover(newPop);
        Mutate(newPop);

        FillPopulationWithRandomValues(newPop, naturSel);

        population = newPop;

        currGenome = 0;

        ResetToCurrentGenome();

    }


    private void Mutate(NNet[] newPopulation)
    {

        for (int i = 0; i < naturSel; i++)
        {

            for (int c = 0; c < newPopulation[i].weights.Count; c++)
            {

                if (Random.Range(0.0f, 1.0f) < mutRate)
                {
                    newPopulation[i].weights[c] = MutateMatrix(newPopulation[i].weights[c]);
                }

            }

        }

    }

    Matrix<float> MutateMatrix(Matrix<float> A)
    {

        int randomPoints = Random.Range(1, (A.RowCount * A.ColumnCount) / 7);

        Matrix<float> C = A;

        for (int i = 0; i < randomPoints; i++)
        {
            int randomColumn = Random.Range(0, C.ColumnCount);
            int randomRow = Random.Range(0, C.RowCount);

            C[randomRow, randomColumn] = Mathf.Clamp(C[randomRow, randomColumn] + Random.Range(-1f, 1f), -1f, 1f);
        }

        return C;

    }

    private void Crossover(NNet[] newPopulation)
    {
        for (int i = 0; i < numToCross; i += 2)
        {
            int AIndex = i;
            int BIndex = i + 1;

            if (genePool.Count >= 1)
            {
                for (int l = 0; l < 100; l++)
                {
                    AIndex = genePool[Random.Range(0, genePool.Count)];
                    BIndex = genePool[Random.Range(0, genePool.Count)];

                    if (AIndex != BIndex)
                        break;
                }
            }

            NNet Child1 = new NNet();
            NNet Child2 = new NNet();

            Child1.Initialise(controller.LAYERS, controller.NEURONS);
            Child2.Initialise(controller.LAYERS, controller.NEURONS);

            Child1.fitness = 0;
            Child2.fitness = 0;


            for (int w = 0; w < Child1.weights.Count; w++)
            {

                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    Child1.weights[w] = population[AIndex].weights[w];
                    Child2.weights[w] = population[BIndex].weights[w];
                }
                else
                {
                    Child2.weights[w] = population[AIndex].weights[w];
                    Child1.weights[w] = population[BIndex].weights[w];
                }

            }


            for (int w = 0; w < Child1.biases.Count; w++)
            {

                if (Random.Range(0.0f, 1.0f) < 0.5f)
                {
                    Child1.biases[w] = population[AIndex].biases[w];
                    Child2.biases[w] = population[BIndex].biases[w];
                }
                else
                {
                    Child2.biases[w] = population[AIndex].biases[w];
                    Child1.biases[w] = population[BIndex].biases[w];
                }

            }

            newPopulation[naturSel] = Child1;
            naturSel++;

            newPopulation[naturSel] = Child2;
            naturSel++;

        }
    }

    private NNet[] PickBestPopulation()
    {

        NNet[] newPopulation = new NNet[initPop];

        for (int i = 0; i < bestAgentSel; i++)
        {
            newPopulation[naturSel] = population[i].InitialiseCopy(controller.LAYERS, controller.NEURONS);
            newPopulation[naturSel].fitness = 0;
            naturSel++;

            int f = Mathf.RoundToInt(population[i].fitness * 10);

            for (int c = 0; c < f; c++)
            {
                genePool.Add(i);
            }

        }

        for (int i = 0; i < worstAgentSel; i++)
        {
            int last = population.Length - 1;
            last -= i;

            int f = Mathf.RoundToInt(population[last].fitness * 10);

            for (int c = 0; c < f; c++)
            {
                genePool.Add(last);
            }

        }

        return newPopulation;

    }

    private void SortPopulation()
    {
        for (int i = 0; i < population.Length; i++)
        {
            for (int j = i; j < population.Length; j++)
            {
                if (population[i].fitness < population[j].fitness)
                {
                    NNet temp = population[i];
                    population[i] = population[j];
                    population[j] = temp;
                }
            }
        }

    }
}
