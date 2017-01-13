using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var data = new List<Tuple<double[], double[]>>();
            var x1 = new double[] { 0.0, 0.0 };
            var x2 = new double[] { 0.0, 1.0 };
            var x3 = new double[] { 1.0, 0.0 };
            var x4 = new double[] { 1.0, 1.0 };

            var y1 = new double[] { 0.0 };
            var y2 = new double[] { 1.0 };
            var y3 = new double[] { 1.0 };
            var y4 = new double[] { 0.0 };

            var item1 = new Tuple<double[], double[]>(x1, y1);
            var item2 = new Tuple<double[], double[]>(x2, y2);
            var item3 = new Tuple<double[], double[]>(x3, y3);
            var item4 = new Tuple<double[], double[]>(x4, y4);

            data.Add(item1); data.Add(item2); data.Add(item3); data.Add(item4);

            ANN n = new ANN(2, 2, 1, 0.5);

            n.Train(data, 100);

            var predict = n.Predict(x3);

            Console.Read();
        }
    }
}

