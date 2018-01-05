using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Collections;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Numerics;
using System.Diagnostics;

namespace ArtificialIntelligence.API.Services
{
    public class Predict4Cntk
    {
        public IEnumerable<string> AnalyzeImageWithTensor(byte[] image)
        {
            throw new NotImplementedException();
        }
    }

    public static class CntkHelpers
    {
        public static Function CreateLogisticModel(Variable input, int numOutputClasses)
        {
            if (input==null)
                input = new Parameter(new int[] { numOutputClasses }, DataType.Float, 0);

            Parameter bias = new Parameter(new int[] { numOutputClasses }, DataType.Float, 0);
            Parameter weights = new Parameter(new int[] { input.Shape[0], numOutputClasses }, DataType.Float,
              CNTKLib.GlorotUniformInitializer(
                CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 1));
            var z = CNTKLib.Plus(bias, CNTKLib.Times(weights, input));
            Function logisticClassifier = CNTKLib.Sigmoid(z, "LogisticClassifier");
            return logisticClassifier;
        }

        private static readonly DeviceDescriptor CPUDeviceDescriptor = DeviceDescriptor.CPUDevice;

        public static Function LoadModel(string modelFilePath)
        {

            if (!File.Exists(modelFilePath))
            {
                throw new FileNotFoundException(
                    modelFilePath,
                    $"Error: The model '{modelFilePath}' does not exist.");
            }

            var modelFunction = Function.Load(modelFilePath, CPUDeviceDescriptor);

            // Get input variable. The model has only one single input.
            //return modelFunction.Arguments.Single();
            return modelFunction;
        }

        public static Tensor<float> ConvertImageToTensorData(string imageFilename, Variable input)
        {
            return ConvertImageToTensorData((Bitmap)Image.FromFile(imageFilename), input);
        }

        public static Tensor<float> ConvertImageToTensorData(Bitmap image, Variable input)
        {
            int channels = input.Shape.Dimensions[0];
            int width = input.Shape.Dimensions[1];
            int height = input.Shape.Dimensions[2];

            //image = ResizeImage(image, new Size(width, height));
            image = ResizeImage(image, new Size(width, height));

            Tensor<float> imageData = new DenseTensor<float>(new[] { width, height, channels }, false); // false: row-major; true: column-major; CNTK uses ColumnMajor layout

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Color color = image.GetPixel(x, y);
                    // float pixelValue = (color.R + color.G + color.B) / 3;
                    imageData[x, y, 0] = NormalizeRGB(color.R);
                    imageData[x, y, 1] = NormalizeRGB(color.G);
                    imageData[x, y, 2] = NormalizeRGB(color.B);
                }
            }

            return imageData;
        }

        private static float NormalizeRGB(float value)
        {
            const float maxValue = 255f;
            const float mean = 2f / maxValue;
            return mean * (value - maxValue) + 1f;
        }

        private static Bitmap ResizeImage(Bitmap imgToResize, Size size)
        {
            Bitmap b = new Bitmap(size.Width, size.Height);
            using (Graphics g = Graphics.FromImage(b))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
            }
            return b;
        }

        public static IEnumerable<float> Evaluate(Tensor<float> imageData, Function modelFunction)
        {
            try
            {
                // Get output variable
                Variable output = modelFunction.Output;
                var input = modelFunction.Arguments.Single();

                var inputDataMap = new Dictionary<Variable, Value>();
                var outputDataMap = new Dictionary<Variable, Value>();

                // Create input data map
                Value inputVal = Value.CreateBatch(input.Shape, imageData, CPUDeviceDescriptor);
                inputDataMap.Add(input, inputVal);

                // Create output data map
                outputDataMap.Add(output, null);

                // Start evaluation on the device
                modelFunction.Evaluate(inputDataMap, outputDataMap, CPUDeviceDescriptor);

                // Get evaluate result as dense output
                Value outputVal = outputDataMap[output];

                // The model has only one single output - a list of 10 floats
                // representing the likelihood of that index being the digit
                IList<float> outputData = outputVal.GetDenseData<float>(output).Single();

                return outputData;
            }
            catch (Exception ex)
            {
                Debug.WriteLine(ex.ToString());
                return Enumerable.Empty<float>();
            }
        }
    }
}
