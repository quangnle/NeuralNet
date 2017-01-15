using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public class NNLayer
    {
        public int NumberOfNeurons { get; set; }

        public double[] BiasWeights { get; set; }

        public double[] Input { get; set; }

        public double[] Output { get; set; }

        public double[] Delta { get; set; }

        public double[,] Weights { get; set; }

        public NNLayer(int nInputs, int nNeurons)
        {
            this.BiasWeights = new double[nNeurons];
            this.Delta = new double[nNeurons];
            this.Output = new double[nNeurons];
            this.Weights = new double[nInputs, nNeurons];
            this.NumberOfNeurons = nNeurons;

            // initialize weights
            for (int i = 0; i < nNeurons; i++)
            {
                BiasWeights[i] = ANN.Rand();
                for (int j = 0; j < nInputs; j++)
                {
                    Weights[j, i] = ANN.Rand();
                }
            }

        }

        public void UpdateOutput()
        {   
            for (int i = 0; i < NumberOfNeurons; i++)
            {
                var z = BiasWeights[i];
                for (int j = 0; j < Input.Length; j++)
                {
                    z += Input[j] * Weights[j, i];
                }

                Output[i] = 1.0 / (1.0 + Math.Exp(-z));
            }
        }
    }
}
