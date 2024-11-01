using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using MathNet.Numerics.LinearAlgebra;
using System;

using Random = UnityEngine.Random;

public class NNet : MonoBehaviour
{
    public Matrix<float> inLay = Matrix<float>.Build.Dense(1, 5);

    public List<Matrix<float>> hidLay = new List<Matrix<float>>();

    public Matrix<float> outLay = Matrix<float>.Build.Dense(1, 2);

    public List<Matrix<float>> weights = new List<Matrix<float>>();

    public List<float> biases = new List<float>();

    public float fitness;

    public void Initialise(int hidLayC, int hidNeurC)
    {
        hidLay.Clear();
        inLay.Clear();
        outLay.Clear();
        weights.Clear();
        biases.Clear();

        for (int i = 0; i < hidLayC + 1; i++)
        {

            Matrix<float> f = Matrix<float>.Build.Dense(1, hidNeurC);

            hidLay.Add(f);

            biases.Add(Random.Range(-1f, 1f));

            if (i == 0)
            {
                Matrix<float> inputToH1 = Matrix<float>.Build.Dense(5, hidNeurC);
                weights.Add(inputToH1);
            }

            Matrix<float> HiddenToHidden = Matrix<float>.Build.Dense(hidNeurC, hidNeurC);
            weights.Add(HiddenToHidden);

            Matrix<float> OutputWeight = Matrix<float>.Build.Dense(hidNeurC, 2);
            weights.Add(OutputWeight);
            biases.Add(Random.Range(-1f, 1f));

            RandomiseWeights();

        }
    }

    public NNet InitialiseCopy(int hiddenLayerCount, int hiddenNeuronCount)
    {
        NNet n = new NNet();

        List<Matrix<float>> newWeights = new List<Matrix<float>>();

        for (int i = 0; i < this.weights.Count; i++)
        {
            Matrix<float> currentWeight = Matrix<float>.Build.Dense(weights[i].RowCount, weights[i].ColumnCount);

            for (int x = 0; x < currentWeight.RowCount; x++)
            {
                for (int y = 0; y < currentWeight.ColumnCount; y++)
                {
                    currentWeight[x, y] = weights[i][x, y];
                }
            }

            newWeights.Add(currentWeight);
        }

        List<float> newBiases = new List<float>();

        newBiases.AddRange(biases);

        n.weights = newWeights;
        n.biases = newBiases;

        n.InitialiseHidden(hiddenLayerCount, hiddenNeuronCount);

        return n;
    }

    private void RandomiseWeights()
    {

            for (int i = 0; i < weights.Count; i++)
            {

                for (int x = 0; x < weights[i].RowCount; x++)
                {

                    for (int y = 0; y < weights[i].ColumnCount; y++)
                    {

                        weights[i][x, y] = Random.Range(-1f, 1f);

                    }

                }

            }

        }

    public void InitialiseHidden(int hiddenLayerCount, int hiddenNeuronCount)
    {
        inLay.Clear();
        hidLay.Clear();
        outLay.Clear();

        for (int i = 0; i < hiddenLayerCount + 1; i++)
        {
            Matrix<float> newHiddenLayer = Matrix<float>.Build.Dense(1, hiddenNeuronCount);
            hidLay.Add(newHiddenLayer);
        }

    }

    public (float, float) RunNetwork(float a, float b, float c, float d, float e)
        {
            inLay[0, 0] = a;
            inLay[0, 1] = b;
            inLay[0, 2] = c;
            inLay[0, 3] = d;
            inLay[0, 4] = e;

            inLay = inLay.PointwiseTanh();

            hidLay[0] = ((inLay * weights[0]) + biases[0]).PointwiseTanh();

            for (int i = 1; i < hidLay.Count; i++)
            {
                hidLay[i] = ((hidLay[i - 1] * weights[i]) + biases[i]).PointwiseTanh();
            }

            outLay = ((hidLay[hidLay.Count - 1] * weights[weights.Count - 1]) + biases[biases.Count - 1]).PointwiseTanh();

            //First output is acceleration and second output is steering
            return (Sigmoid(outLay[0, 0]), (float)Math.Tanh(outLay[0, 1]));
        }


        private float Sigmoid(float s)
        {
            return (1 / (1 + Mathf.Exp(-s)));
        }
}
