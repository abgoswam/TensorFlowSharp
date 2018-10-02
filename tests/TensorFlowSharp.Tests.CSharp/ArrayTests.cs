using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using TensorFlow;
using Xunit;
using ExampleCommon;

namespace TensorFlowSharp.Tests.CSharp
{
    public class ArrayTests
    {
        [Fact]
        public void FrozenModelLoad()
        {
            // Works with frozen models
            string modelFile = @"models\frozen_saved_model.pb";

            // Construct an in-memory graph from the serialized form.
            var graph = new TFGraph();

            // Load the serialized GraphDef from a file.
            var model = File.ReadAllBytes(modelFile);
            graph.Import(model, "");

            using (var session = new TFSession(graph))
            {
                TFTensor tensor = new float[,] { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };

                var runner = session.GetRunner();
                runner.AddInput(graph["I"][0], tensor).Fetch(graph["O"][0]);
                var output = runner.Run();
            }
        }

        [Fact]
        public void SavedModelLoad()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"models";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
            TFTensor tensor = new float[,] { { 1.0f, 2.0f, 3.0f }, { 4.0f, 5.0f, 6.0f } };

            var runner = session.GetRunner();
            runner.AddInput(graph["I"][0], tensor).Fetch(graph["O"][0]);
            var output = runner.Run();

            var result = output[0];
            var rshape = result.Shape;
            var val = (float[,])result.GetValue(jagged: false);
            Assert.True(val[0, 0] == 31);
        }

        [Fact(Skip ="Not working as yet")]
        public void FreezeSavedModel()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"models";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);

            // can we freeze graph ?
            var items = graph.GetEnumerator();
            foreach(var op in items)
            {
                //if (op.OpType == "Variable")
                //{
                //    var op_clone = new TFOperation();
                //    var attr = op.GetAttributeMetadata("dtype");
                //    var value = op.GetAttributeMetadata("value");
                //}
            }
        }

        [Fact]
        public void RetrainSavedModel()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"lr_model";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);

            var features = new float[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26,
                    166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253,
                    225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253,
                    253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198,
                    182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0,
                    43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18, 126, 136, 175, 26,
                    166, 255, 247, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253, 253, 253, 253, 253,
                    225, 172, 253, 242, 195, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253, 253, 253, 253, 253,
                    253, 251, 93, 82, 82, 56, 39, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253, 253, 253, 253, 253, 198,
                    182, 247, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 80, 156, 107, 253, 253, 205, 11, 0,
                    43, 154, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249, 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130, 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148, 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114, 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66, 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171, 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172, 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };

            var label = new float[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            var runner = session.GetRunner();
            runner.AddInput("Features", TFTensor.FromBuffer(new TFShape(new long[] { 2, 784}), features, 0, 2*784));
            runner.AddInput("Label", TFTensor.FromBuffer(new TFShape(new long[] { 2, 10 }), label, 0, 2*10));
            runner.AddInput("SGDOptimizer/learning_rate", new TFTensor(0.001f));
            runner.Fetch("Loss");
            runner.AddTarget("SGDOptimizer");

            var tensor = runner.Run();
            var loss = tensor.Length > 0 ? (float)tensor[0].GetValue() : 0.0f;
            //var metric = tensor.Length > 1 ? (float)tensor[1].GetValue() : 0.0f;
        }

        [Fact]
        public void RetrainSavedModelCifar_1()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"model_two_layer_convnet-e05d7e52-601e-491a-a408-58e71fc796c5";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
            var x_tensor = ImageUtil2.CreateTensorFromImageFile("/tmp/demo_32_32_3.jpg");
            var labels  = new long[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var y_tensor = TFTensor.FromBuffer(new TFShape(new long[] { 1, 10 }), labels, 0, 1 * 10);

            var runner = session.GetRunner();
            runner.AddInput("AG_X:0", x_tensor);
            runner.AddInput("AG_y:0", y_tensor);
            runner.Fetch("add_1:0");
            var output = runner.Run();

            var result = output[0];
            var rshape = result.Shape;
            var val = (float[,])result.GetValue(jagged: false);
            Assert.True(val.Length == 10);
        }

        [Fact]
        public void RetrainSavedModelCifar_2()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"model_two_layer_convnet-e05d7e52-601e-491a-a408-58e71fc796c5";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
            var x_tensor = ImageUtil2.CreateTensorFromImageFile("/tmp/demo_32_32_3.jpg");
            var labels = new long[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var y_tensor = TFTensor.FromBuffer(new TFShape(new long[] { 1, 10 }), labels, 0, 1 * 10);

            var runner = session.GetRunner();
            runner.AddInput("AG_X:0", x_tensor);
            runner.AddInput("AG_y:0", y_tensor);
            runner.Fetch("add_1:0");
            var output = runner.Run();

            var result = output[0];
            var rshape = result.Shape;
            var val = (float[,])result.GetValue(jagged: false);
            Assert.True(val.Length == 10);
        }

        [Fact]
        public void MultiD()
        {
            /// The same array with dimensions specified.
            int[,] array2Da = new int[4, 2] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } };

            int[][] jaggedArray2 = new int[][] {
                new int[] {1,3,5,7,9},
                new int[] {0,2,4,6},
                new int[] {11,22}
            };

            Assert.True(jaggedArray2[0].Length == 5);
        }

        [Fact]
        public void SavedModelLoadCifar()
        {
            var sessionOptions = new TFSessionOptions();
            var exportDir = @"model_cifar10";
            var tags = new string[] { "serve" };
            var graph = new TFGraph();
            var metaGraphDef = new TFBuffer();

            var session = TFSession.FromSavedModel(sessionOptions, null, exportDir, tags, graph, metaGraphDef);
            var tensor = ImageUtil2.CreateTensorFromImageFile("/tmp/demo_32_32_3.jpg");

            var runner = session.GetRunner();
            runner.AddInput(graph["Input"][0], tensor).Fetch(graph["Output"][0]);
            var output = runner.Run();

            var result = output[0];
            var rshape = result.Shape;
            var val = (float[,])result.GetValue(jagged: false);
            Assert.True(val.Length == 10);
        }

        [Fact]
        public void BasicConstantZerosAndOnes()
        {
            using (var g = new TFGraph())
            using (var s = new TFSession(g))
            {

                // Test Zeros, Ones for n x n shape
                var o = g.Ones(new TFShape(4, 4));
                Assert.NotNull(o);
                Assert.Equal(o.OutputType, TFDataType.Double);

                var z = g.Zeros(new TFShape(4, 4));
                Assert.NotNull(z);
                Assert.Equal(z.OutputType, TFDataType.Double);

                var r = g.RandomNormal(new TFShape(4, 4));
                Assert.NotNull(r);
                Assert.Equal(r.OutputType, TFDataType.Double);

                var res1 = s.GetRunner().Run(g.Mul(o, r));
                Assert.NotNull(res1);
                Assert.Equal(res1.TensorType, TFDataType.Double);
                Assert.Equal(res1.NumDims, 2);
                Assert.Equal(res1.Shape[0], 4);
                Assert.Equal(res1.Shape[1], 4);
                Assert.Equal(res1.ToString(), "[4x4]");

                var matval1 = res1.GetValue();
                Assert.NotNull(matval1);
                Assert.IsType(typeof(double[,]), matval1);
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])matval1)[i, j]);
                    }
                }

                var res2 = s.GetRunner().Run(g.Mul(g.Mul(o, r), z));
                Assert.NotNull(res2);
                Assert.Equal(res2.TensorType, TFDataType.Double);
                Assert.Equal(res2.NumDims, 2);
                Assert.Equal(res2.Shape[0], 4);
                Assert.Equal(res2.Shape[1], 4);
                Assert.Equal(res2.ToString(), "[4x4]");

                var matval2 = res2.GetValue();
                Assert.NotNull(matval2);
                Assert.IsType(typeof(double[,]), matval2);

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        Assert.NotNull(((double[,])matval2)[i, j]);
                        Assert.Equal(((double[,])matval2)[i, j], 0.0);
                    }
                }
            }
        }

#if false
		[Fact]
		public void BasicConstantsOnSymmetricalShapes ()
		{
			using (var g = new TFGraph ())
			using (var s = new TFSession (g)) {
				//build some test vectors
				var o = g.Ones (new TFShape (4, 4));
				var z = g.Zeros (new TFShape (4, 4));
				var r = g.RandomNormal (new TFShape (4, 4));
				var matval = s.GetRunner ().Run (g.Mul (o, r)).GetValue ();
				var matvalzero = s.GetRunner ().Run (g.Mul (g.Mul (o, r), z)).GetValue ();

				var co = g.Constant (1.0, new TFShape (4, 4), TFDataType.Double);
				var cz = g.Constant (0.0, new TFShape (4, 4), TFDataType.Double);
				var res1 = s.GetRunner ().Run (g.Mul (co, r));

				Assert.NotNull (res1);
				Assert.Equal (res1.TensorType, TFDataType.Double);
				Assert.Equal (res1.NumDims, 2);
				Assert.Equal (res1.Shape [0], 4);
				Assert.Equal (res1.Shape [1], 4);
				Assert.Equal (res1.ToString (), "[4x4]");

				var cmatval1 = res1.GetValue ();
				Assert.NotNull (cmatval1);
				Assert.IsType (typeof (double [,]), cmatval1 );
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])cmatval1) [i, j]);
					}
				}

				var cres2 = s.GetRunner ().Run (g.Mul (g.Mul (co, r), cz));

				Assert.NotNull (cres2);
				Assert.Equal (cres2.TensorType, TFDataType.Double);
				Assert.Equal (cres2.NumDims, 2);
				Assert.Equal (cres2.Shape [0], 4);
				Assert.Equal (cres2.Shape [1], 4);
				Assert.Equal (cres2.ToString (), "[4x4]");

				var cmatval2 = cres2.GetValue ();
				Assert.NotNull (cmatval2);
				Assert.IsType (typeof (double [,]), cmatval2);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 4; j++) {
						Assert.NotNull (((double [,])cmatval2) [i, j]);
						Assert.Equal (((double [,])matvalzero) [i, j], ((double [,])cmatval2) [i, j]);
					}
				}
			}
		}

		[Fact]
		public void BasicConstantsUnSymmetrical ()
		{
			using (var g = new TFGraph ())
			using (var s = new TFSession (g)) {
				var o = g.Ones (new TFShape (4, 3));
				Assert.NotNull (o);
				Assert.Equal (o.OutputType, TFDataType.Double);

				var r = g.RandomNormal (new TFShape (3, 5));
				Assert.NotNull (o);
				Assert.Equal (o.OutputType, TFDataType.Double);

				//expect incompatible shapes
				Assert.Throws<TFException> (() => s.GetRunner ().Run (g.Mul (o, r)));

				var res = s.GetRunner ().Run (g.MatMul (o, r));
				Assert.NotNull (res);
				Assert.Equal (res.TensorType, TFDataType.Double);
				Assert.Equal (res.NumDims, 2);
				Assert.Equal (res.Shape [0], 4);
				Assert.Equal (res.Shape [1], 5);

				double [,] val = (double [,])res.GetValue ();
				Assert.NotNull (val);
				Assert.IsType (typeof (double [,]), val);
				for (int i = 0; i < 4; i++) {
					for (int j = 0; j < 5; j++) {
						Assert.NotNull (((double [,])val) [i, j]);
					}
				}
			}
		}
#endif

        private static IEnumerable<object[]> stackData()
        {
            // Example from https://www.tensorflow.org/api_docs/python/tf/stack

            // 'x' is [1, 4]
            // 'y' is [2, 5]
            // 'z' is [3, 6]

            double[] x = { 1, 4 };
            double[] y = { 2, 5 };
            double[] z = { 3, 6 };

            // stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  // Pack along first dim.
            // stack([x, y, z], axis= 1) => [[1, 2, 3], [4, 5, 6]]

            yield return new object[] { x, y, z, null,  new double[,] { { 1, 4 },
                                                                        { 2, 5 },
                                                                        { 3, 6 } } };

            yield return new object[] { x, y, z, 1, new double[,] { { 1, 2, 3 },
                                                                    { 4, 5, 6 } } };
        }

        [Theory]
        [MemberData(nameof(stackData))]
        public void Should_Stack(double[] x, double[] y, double[] z, int? axis, double[,] expected)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                var a = graph.Placeholder(TFDataType.Double, new TFShape(2));
                var b = graph.Placeholder(TFDataType.Double, new TFShape(2));
                var c = graph.Placeholder(TFDataType.Double, new TFShape(2));

                TFOutput r = graph.Stack(new[] { a, b, c }, axis: axis);

                TFTensor[] result = session.Run(new[] { a, b, c }, new TFTensor[] { x, y, z }, new[] { r });

                double[,] actual = (double[,])result[0].GetValue();
                TestUtils.MatrixEqual(expected, actual, precision: 10);
            }
        }

        private static IEnumerable<object[]> rangeData()
        {
            double[] x = { 1, 4 };
            double[] y = { 2, 5 };
            double[] z = { 3, 6 };

            // 'start' is 3
            // 'limit' is 18
            // 'delta' is 3
            //  tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

            // 'start' is 3
            // 'limit' is 1
            // 'delta' is -0.5
            //  tf.range(start, limit, delta) ==> [3, 2.5, 2, 1.5]

            // 'limit' is 5
            //  tf.range(limit) ==> [0, 1, 2, 3, 4]

            yield return new object[] { 3, 18, 3, new int[] { 3, 6, 9, 12, 15 } };
            yield return new object[] { 3, 1, -0.5, new double[] { 3, 2.5, 2, 1.5 } };
            yield return new object[] { 3, 1, -0.5f, new float[] { 3, 2.5f, 2, 1.5f } };
            yield return new object[] { null, 5, null, new int[] { 0, 1, 2, 3, 4 } };
            yield return new object[] { null, 5f, null, new float[] { 0, 1, 2, 3, 4f } };
        }

        [Theory]
        [MemberData(nameof(rangeData))]
        public void Should_Range(object start, object limit, object delta, object expected)
        {
            // Examples from https://www.tensorflow.org/api_docs/python/tf/range

            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                TFOutput tstart = graph.Placeholder(start == null ? TFDataType.Int32 : TensorTypeFromType(start.GetType()));
                TFOutput tlimit = graph.Placeholder(limit == null ? TFDataType.Int32 : TensorTypeFromType(limit.GetType()));
                TFOutput tdelta = graph.Placeholder(delta == null ? TFDataType.Int32 : TensorTypeFromType(delta.GetType()));

                TFTensor[] result;
                if (start == null && delta == null)
                {
                    TFOutput r = graph.Range(tlimit);
                    result = session.Run(new[] { tlimit }, new TFTensor[] { TensorFromObject(limit) }, new[] { r });
                }
                else
                {
                    TFOutput r = graph.Range(tstart, (Nullable<TFOutput>)tlimit, (Nullable<TFOutput>)tdelta);
                    result = session.Run(new[] { tstart, tlimit, tdelta },
                        new TFTensor[] { TensorFromObject(start), TensorFromObject(limit), TensorFromObject(delta) },
                        new[] { r });
                }

                Array actual = (Array)result[0].GetValue();
                TestUtils.MatrixEqual((Array)expected, actual, precision: 10);
            }
        }


        private static IEnumerable<object[]> transposeData()
        {
            yield return new object[] { new double [,] { { 1, 2 },
                                                          { 3, 4 } }};
            yield return new object[] { new double [,] { { 1, 2, 3 },
                                                          { 4, 5, 6} }};
            yield return new object[] { new double [,] { { 1 },
                                                          { 3 } }};
            yield return new object[] { new double[,] { { 1, 3 } } };
        }

        [Theory]
        [MemberData(nameof(transposeData))]
        public void nShould_Transpose(double[,] x)
        {
            using (var graph = new TFGraph())
            using (var session = new TFSession(graph))
            {
                TFOutput a = graph.Placeholder(TFDataType.Double, new TFShape(2));

                TFOutput r = graph.Transpose(a);

                TFTensor[] result = session.Run(new[] { a }, new TFTensor[] { x }, new[] { r });

                double[,] actual = (double[,])result[0].GetValue();
                double[,] expected = new double[x.GetLength(1), x.GetLength(0)];
                for (int i = 0; i < expected.GetLength(0); i++)
                {
                    for (int j = 0; j < expected.GetLength(1); j++)
                    {
                        expected[i, j] = x[j, i];
                    }
                }

                TestUtils.MatrixEqual(expected, actual, precision: 10);
            }
        }



        public static TFDataType TensorTypeFromType(Type type)
        {
            if (type == typeof(float))
                return TFDataType.Float;
            if (type == typeof(double))
                return TFDataType.Double;
            if (type == typeof(int))
                return TFDataType.Int32;
            if (type == typeof(byte))
                return TFDataType.UInt8;
            if (type == typeof(short))
                return TFDataType.Int16;
            if (type == typeof(sbyte))
                return TFDataType.Int8;
            if (type == typeof(String))
                return TFDataType.String;
            if (type == typeof(bool))
                return TFDataType.Bool;
            if (type == typeof(long))
                return TFDataType.Int64;
            if (type == typeof(ushort))
                return TFDataType.UInt16;
            if (type == typeof(Complex))
                return TFDataType.Complex128;

            throw new ArgumentOutOfRangeException("type");
        }

        public static TFTensor TensorFromObject(object obj)
        {
            Type type = obj.GetType();
            if (type == typeof(float))
                return new TFTensor((float)obj);
            if (type == typeof(double))
                return new TFTensor((double)obj);
            if (type == typeof(int))
                return new TFTensor((int)obj);
            if (type == typeof(byte))
                return new TFTensor((byte)obj);
            if (type == typeof(short))
                return new TFTensor((short)obj);
            if (type == typeof(sbyte))
                return new TFTensor((sbyte)obj);
            if (type == typeof(String))
                throw new NotImplementedException();
            if (type == typeof(bool))
                return new TFTensor((bool)obj);
            if (type == typeof(long))
                return new TFTensor((long)obj);
            if (type == typeof(ushort))
                return new TFTensor((ushort)obj);
            if (type == typeof(Complex))
                return new TFTensor((Complex)obj);

            throw new ArgumentOutOfRangeException("type");
        }
    }
}
