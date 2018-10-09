using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace NeuralNetwork
{
    class Network_2
    {
        public static void Main_2()
        {
            var learning_rate = 0.01;
            double Epochs = 200;
            int BatchSize = 128;

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

                var weight = graph.VariableV2(new TFShape(new long[] { InputSize, OutputSize }), TFDataType.Double, operName: "weight_out");
                var bias = graph.VariableV2(new TFShape(new long[] { OutputSize }), TFDataType.Double, operName: "bias_out");

                var initWeight = graph.Assign(weight, graph.RandomNormal(graph.GetTensorShape(weight)));
                var initBias = graph.Assign(bias, graph.RandomNormal(graph.GetTensorShape(bias)));

                var pred = graph.Add(graph.MatMul(X, weight), bias);

                var pred_shape = graph.GetShape(pred);
                var Y_shape = graph.GetShape(Y);

                //TFOutput loss = graph.ReduceMean(graph.SigmoidCrossEntropyWithLogits(Y, pred));
                TFOutput loss = graph.ReduceSum(pred);
                var loss_shape = graph.GetShape(loss);

                var grad = graph.AddGradients(new TFOutput[] { loss }, new TFOutput[] { weight, bias });
                var optimize = new[]
                {
                    graph.AssignSub(weight, graph.Mul(grad[0], graph.Const(learning_rate))).Operation,
                    graph.AssignSub(bias, graph.Mul(grad[1], graph.Const(learning_rate))).Operation
                };

                var sess = new TFSession(graph);
                sess.GetRunner().AddTarget(initWeight.Operation, initBias.Operation).Run();

                for (int i = 0; i < Epochs; i++)
                {
                    var result = sess.GetRunner()
                        .AddInput(X, input)
                        .AddInput(Y, output)
                        .AddTarget(optimize)
                        .Fetch(loss, weight, bias)
                        .Run();

                    Console.WriteLine("epoch:{0} loss:{1} W:{2} b:{3}", i, result[0].GetValue(), result[1].GetValue(), result[2].GetValue());
                }
            }
        }

        public static TFOutput Predict(TFOutput x, TFOutput[] w, TFOutput[] b, TFGraph graph)
        {
            TFOutput LayerOut = x;

            for (int i = 0; i < w.Length; i++)
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
