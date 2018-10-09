using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace ExampleTrainPredictGradients
{
    /// <summary>
    /// LinearFit
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            var xList = new List<double>();
            var yList = new List<double>();
            var ran = new Random();
            for (var i = 0; i < 10; i++)
            {
                var num = ran.NextDouble();
                var noise = ran.NextDouble();
                xList.Add(num);
                yList.Add(num * 10 + 10 + noise); // y = 10 * x + 10
            }
            var xData = xList.ToArray();
            var yData = yList.ToArray();
            var learning_rate = 0.01;

            var g = new TFGraph();

            var x = g.Placeholder(TFDataType.Double, new TFShape(xData.Length));
            var y = g.Placeholder(TFDataType.Double, new TFShape(yData.Length));

            var W = g.VariableV2(TFShape.Scalar, TFDataType.Double, operName: "weight");
            var b = g.VariableV2(TFShape.Scalar, TFDataType.Double, operName: "bias");

            var initW = g.Assign(W, g.Const(ran.NextDouble()));
            var initb = g.Assign(b, g.Const(ran.NextDouble()));

            var sx = g.GetShape(x);
            var sW = g.GetShape(W);
            var sb = g.GetShape(b);

            var inter = g.Mul(x, W);
            var sinter = g.GetShape(inter);

            var output = g.Add(inter, b);

            var loss = g.ReduceSum(g.Pow(g.Sub(output, y), g.Const(2.0)));
            var grad = g.AddGradients(new TFOutput[] { loss }, new TFOutput[] { W, b });
            var optimize = new[]
            {
                g.AssignSub(W, g.Mul(grad[0], g.Const(learning_rate))).Operation,
                g.AssignSub(b, g.Mul(grad[1], g.Const(learning_rate))).Operation
            };

            var sess = new TFSession(g);

            sess.GetRunner().AddTarget(initW.Operation, initb.Operation).Run();

            for (var i = 0; i < 1000; i++)
            {
                var result = sess.GetRunner()
                    .AddInput(x, xData)
                    .AddInput(y, yData)
                    .AddTarget(optimize)
                    .Fetch(loss, W, b).Run();

                Console.WriteLine("loss: {0} W:{1} b:{2}", result[0].GetValue(),
                    result[1].GetValue(), result[2].GetValue());
            }
        }
    }
}
