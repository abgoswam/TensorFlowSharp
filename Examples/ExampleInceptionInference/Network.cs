using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace NeuralNetwork
{
    class Network
    {
        public static void Main_1()
        {
            var LearningRate = 0.01;
            double Epochs = 200;
            double DisplaySteps = 10;
            int BatchSize = 128;
            int[] NodeSize = new int[] { 1 };

            double[][] input = new double[][] {
                new double[] {1,3,5,7,9},
                new double[] {0,2,4,6,1},
                new double[] {1,2,3,4,5}
            };

            double[][] output = new double[][] {
                new double[] {1,0,0},
                new double[] {0,1,0},
                new double[] {0,0,1}
            };

            int InputSize = input[0].Length;
            int OutputSize = output[0].Length;


            using (var graph = new TFGraph())
            {
                TFOutput X = graph.Placeholder(TFDataType.Double, new TFShape(new long[] { -1, InputSize }));
                TFOutput Y = graph.Placeholder(TFDataType.Double, new TFShape(new long[] { -1, OutputSize }));

                TFOutput[] weights = new TFOutput[NodeSize.Length + 1];
                TFOutput[] biases = new TFOutput[NodeSize.Length + 1];

                //TFOutput[] weights = new TFOutput[NodeSize.Length];
                //TFOutput[] biases = new TFOutput[NodeSize.Length];

                int prevSize = InputSize;
                for(int i = 0; i < NodeSize.Length; i++)
                {
                    weights[i] = graph.VariableV2(new TFShape(new long[] { prevSize, NodeSize[i]  }), TFDataType.Double, operName: "weight_" + i);
                    biases[i] = graph.VariableV2(new TFShape(new long[] { NodeSize[i] }), TFDataType.Double, operName: "bias_" + i);
                    prevSize = NodeSize[i];

                    var s1 = graph.GetShape(weights[i]);
                    var s2 = graph.GetShape(biases[i]);

                }

                weights[NodeSize.Length] = graph.VariableV2(new TFShape(new long[] { prevSize, OutputSize }), TFDataType.Double, operName: "weight_out");
                biases[NodeSize.Length] = graph.VariableV2(new TFShape(new long[] { OutputSize }), TFDataType.Double, operName: "bias_out");

                TFOutput pred = Predict(X, weights, biases, graph);
                var pred_shape = graph.GetShape(pred);
                var Y_shape = graph.GetShape(Y);

                //TFOutput cost = graph.ReduceMean(graph.SigmoidCrossEntropyWithLogits(Y, pred));
                //TFOutput cost = graph.ReduceSum(graph.Pow(graph.Sub(pred, Y), graph.Const(2.0)));

                TFOutput cost = graph.ReduceSum(pred);
                var cost_shape = graph.GetShape(cost);


                TFOutput[] parameters = new TFOutput[weights.Length + biases.Length];
                weights.CopyTo(parameters, 0);
                biases.CopyTo(parameters, weights.Length);

                //var parameters = new TFOutput[] { weights[0], biases[0] };
                TFOutput[] grad = graph.AddGradients(new TFOutput[] { cost }, parameters);

                TFOperation[] optimize = new TFOperation[parameters.Length];

                for(int i = 0; i < parameters.Length; i++)
                {
                    var eps = graph.Mul(grad[i], graph.Const(LearningRate));
                    optimize[i] = graph.AssignSub(parameters[i], eps).Operation;
                }

                TFSession sess = new TFSession(graph);

                TFOperation[] InitParams = new TFOperation[parameters.Length];

                for (int i = 0; i < parameters.Length; i++)
                {
                    InitParams[i] = graph.Assign(parameters[i], graph.RandomNormal(graph.GetTensorShape(parameters[i]))).Operation;
                }

                sess.GetRunner().AddTarget(InitParams);

                for (int i = 0; i < Epochs; i++)
                {
                    TFTensor result = sess.GetRunner()
                        .AddInput(X, input)
                        .AddInput(Y, output)
                        .AddTarget(optimize)
                        .Fetch(cost)
                        .Run();

                    if (i % DisplaySteps == 0)
                        Console.WriteLine("Epoch - " + i + " | Cost - " + result.GetValue());
                }
            }
        }

        public static TFOutput Predict(TFOutput x, TFOutput[] w, TFOutput[] b, TFGraph graph)
        {
            TFOutput LayerOut = x;

            for(int i = 0; i < w.Length; i++)
            {
                var s1 = graph.GetShape(LayerOut);
                var s2 = graph.GetShape(w[i]);
                var s3 = graph.GetShape(b[i]);

                LayerOut = graph.Add(graph.MatMul(LayerOut, w[i]), b[i]);
                var ls = graph.GetShape(LayerOut);
            }

            return LayerOut;
        }

    }
}
