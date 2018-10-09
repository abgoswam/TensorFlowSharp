using TensorFlow;
using Xunit;

namespace TensorFlowSharp.Tests.CSharp
{
    public class GradientTests
    {
        [Fact]
        public void ShouldAddGradients()
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var x = graph.Const(3.0);

                var y1 = graph.Square(x, "Square1");
                var y2 = graph.Square(y1, "Square2");

                var y3 = graph.Square(y2, "Square3");
                var g = graph.AddGradients(new TFOutput[] { y1, y3 }, new[] { x });

                var r = session.Run(new TFOutput[] { }, new TFTensor[] { }, g);
                var dy = (double)r[0].GetValue();
                Assert.Equal(17502.0, dy);
            }
        }

        [Fact]
        public void ShouldAddGradients_SavedModel()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"model_linear_fit-d0966df2-a553-402d-a998-6b3753d2c3f9";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();
            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);

            var x = graph.Const(3.0);
            var y1 = graph.Square(x, "Square1");
            var y2 = graph.Square(y1, "Square2");
            var y3 = graph.Square(y2, "Square3");
            var g = graph.AddGradients(new TFOutput[] { y1, y3 }, new[] { x });

            var r = session.Run(new TFOutput[] { }, new TFTensor[] { }, g);
            var dy = (double)r[0].GetValue();
            Assert.Equal(17502.0, dy);
        }
    }
}
