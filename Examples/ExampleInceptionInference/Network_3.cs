using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using System.Runtime.CompilerServices;

namespace ExampleInceptionInference
{
    class Network_3
    {
        static public void Assert(bool assert, [CallerMemberName] string caller = null, string message = "")
        {
            if (!assert)
            {
                throw new Exception($"{caller}: {message}");
            }
        }

        static public void Assert(TFStatus status, [CallerMemberName] string caller = null, string message = "")
        {
            if (status.StatusCode != TFCode.Ok)
            {
                throw new Exception($"{caller}: {status.StatusMessage} {message}");
            }
        }

        public static void Main()
        {
            var learning_rate = 0.01f;
            var train_X = new float [] { 6.83f, 4.668f, 8.9f, 7.91f, 5.7f, 8.7f, 3.1f, 2.1f };
            var train_Y = new float[train_X.Length];
            var test_X = new[] { 6.83f, 4.668f, 8.9f };

            for (var i = 0; i < train_X.Length; i++)
            {
                train_Y[i] = train_X[i] * 10;
            }

            var sessionOptions = new TFSessionOptions();

            var exportDir = @"model_linear_fit-d0966df2-a553-402d-a998-6b3753d2c3f9";
            var tags = new string[] { "serve" };
            var g = new TFGraph();
            var metaGraphDef = new TFBuffer();
            var ran = new Random();
            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, g, metaGraphDef);

            var tf_operations = g.GetEnumerator();
            foreach(var op in tf_operations)
            {
                Console.WriteLine(op.Name);
                foreach(var deps in op.ControlOutputs)
                {
                    Console.WriteLine(deps.Name);
                }
                Console.WriteLine("---");
            }

            var status = new TFStatus();
            // Export to GraphDef
            TFBuffer graphDef = new TFBuffer();
            g.ToGraphDef(graphDef, status);
            Assert(status);

            //// Import it again, with a prefix, in a fresh graph
            //using (var graph = new TFGraph())
            //{
            //    using (var options = new TFImportGraphDefOptions())
            //    {
            //        options.SetPrefix("imported");
            //        graph.Import(graphDef, options, status);
            //        Assert(status);
            //    }
            //    graphDef.Dispose();

            //    var scalar = graph["imported/AG_bias"];
            //    Assert(scalar != null);

            //    var restore = graph["imported/save/restore_all"];
            //    Assert(restore != null);

            //    var s = new TFSession(graph, status);
            //    var r1 = s.GetRunner();
            //    var o = r1
            //        .AddTarget("imported/save/restore_all")
            //        .Fetch("imported/AG_bias")
            //        .Run();
            //    var v = o[0].GetValue();
            //}

            var runner = session.GetRunner();
            //var output = runner
            //    .AddInput("AG_X", test_X)
            //    .Fetch("AG_pred")
            //    .Run();
            //var val = output[0].GetValue();

            //var output = runner
            //    .AddTarget("gradients/Shape")
            //    .Run();
            //var val = output[0].GetValue();

            var pred = g["AG_pred"][0];
            var Y = g["AG_Y"][0];

            var W2 = g.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "AG_weight2");
            var b2 = g.VariableV2(TFShape.Scalar, TFDataType.Float, operName: "AG_bias2");

            var initW2 = g.Assign(W2, g.Const((float)ran.NextDouble()), operName: "AG_initW2");
            var initb2 = g.Assign(b2, g.Const((float)ran.NextDouble()), operName: "AG_initb2");
            runner.AddTarget(initW2.Operation, initb2.Operation).Run();

            var pred2 = g.Add(g.Mul(pred, W2), b2, operName: "AG_pred2");
            var loss = g.ReduceSum(g.Pow(g.Sub(pred2, Y), g.Const(2.0f)), operName: "AG_loss2");

            var x = g.Const(3.0);
            var y1 = g.Square(x, "Square1");
            var y2 = g.Square(y1, "Square2");
            var y3 = g.Square(y2, "Square3");
            var k = g.AddGradients(new TFOutput[] { y1, y3 }, new[] { x });
            var r = session.Run(new TFOutput[] { }, new TFTensor[] { }, k);
            var dy = (double)r[0].GetValue();

            var grad = g.AddGradients(new TFOutput[] { loss }, new TFOutput[] { W2, b2 });
            // var grad_W2 = graph.Mul(grad[0], graph.Const(learning_rate), operName: "AG_grad_W2");
            // var grad_b2 = graph.Mul(grad[1], graph.Const(learning_rate), operName: "AG_grad_b2");
            // var update_W2 = graph.AssignSub(W2, grad_W2, operName: "AG_update_W2");
            // var update_b2 = graph.AssignSub(b2, grad_b2, operName: "AG_update_b2");
            // var optimize = new[] { update_W2.Operation, update_b2.Operation };

            for (var i = 0; i < 4; i++)
            {
                runner = session.GetRunner();
                var result = runner
                    .AddInput("AG_X", train_X)
                    .AddInput("AG_Y", train_Y)
                    // .AddTarget(optimize)
                    .Fetch("AG_pred", "AG_loss2").Run();

                //Console.WriteLine("loss: {0} W2:{1} b2:{2}", result[0].GetValue(), result[1].GetValue(), result[2].GetValue());
                var _pred = result[0].GetValue();
                var _loss = result[1].GetValue();
            }
        }
    }
}
