using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class ANN
    {
        private double _learningRate;
        public NNLayer HiddenLayer { get; set; }
        public NNLayer OutputLayer { get; set; }

        public ANN(int nInput, int nHidden, int nOutput, double learnRate)
        {   
            HiddenLayer = new NNLayer(nInput, nHidden);
            OutputLayer = new NNLayer(nHidden, nOutput);
            _learningRate = learnRate;
        }

        private void TrainOnce(double[] input, double[] target)
        {
            // step 1: feed forward
            HiddenLayer.Input = input;
            HiddenLayer.UpdateOutput();
            OutputLayer.Input = HiddenLayer.Output;
            OutputLayer.UpdateOutput();

            // step 2: calculate errors for output layer
            for (int i = 0; i < OutputLayer.NumberOfNeurons; i++)
            {
                OutputLayer.Delta[i] = OutputLayer.Output[i] * (1 - OutputLayer.Output[i]) * (target[i] - OutputLayer.Output[i]);
            }

            // step 3: update weights to output neurons

            for (int i = 0; i < HiddenLayer.NumberOfNeurons; i++)
            {
                for (int j = 0; j < OutputLayer.NumberOfNeurons; j++)
                {
                    OutputLayer.Weights[i, j] += _learningRate * OutputLayer.Delta[j] * HiddenLayer.Output[i];
                }
            }

            // step 4: calculate errors for hidden layer
            for (int i = 0; i < HiddenLayer.NumberOfNeurons; i++)
            {
                var deriv = HiddenLayer.Output[i] * (1 - HiddenLayer.Output[i]);
                var delta = 0.0;
                for (int j = 0; j < OutputLayer.NumberOfNeurons; j++)
                {
                    delta += OutputLayer.Delta[j] * OutputLayer.Weights[i, j];
                }
                HiddenLayer.Delta[i] = delta * deriv;
            }

            // step 5: update weights for hidden layer
            for (int i = 0; i < HiddenLayer.Input.Length; i++)
            {
                for (int j = 0; j < HiddenLayer.NumberOfNeurons; j++)
                {
                    HiddenLayer.Weights[i, j] += _learningRate * HiddenLayer.Delta[i] * input[j];
                }
            }
        }

        public void Train(List<Tuple<double[], double[]>> traindata, int times)
        {
            for (int i = 0; i < times; i++)
            {
                for (int j = 0; j < traindata.Count; j++)
                {
                    var input = traindata[j].Item1;
                    var target = traindata[j].Item2;

                    TrainOnce(input, target);
                }
            }
        }

        public double[] Predict(double[] inputs)
        {
            HiddenLayer.Input = inputs;
            HiddenLayer.UpdateOutput();
            OutputLayer.Input = HiddenLayer.Output;
            OutputLayer.UpdateOutput();

            return OutputLayer.Output;
        }
    }
}