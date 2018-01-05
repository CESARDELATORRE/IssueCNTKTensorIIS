﻿using CNTK;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialIntelligence.API.CNTK.Services.ComputerVision.Model
{
    public interface IClassifier
    {
        IEnumerable<LabelConfidence> ClassifyImage(byte[] image);
    }

    public class LabelConfidence
    {
        public float Probability { get; set; }
        public string Label { get; set; }
    }

    public class CNTKModelPredictionSettings
    {
        public string BinFolder { get; set; }
        public string ModelFilename { get; set; }
        public string LabelsFilename { get; set; }
        public float Threshold { get; set; }
    }

    static class CNTKExtension
    {
        /// <summary>
        /// Launches a task that performs evaluation on the computation graph defined by 'function', using provided 'input'
        /// and stores the results in the 'outputs' map.
        /// It is implemented as an extension method of the class Function.
        /// </summary>
        /// <param name="function"> The function representing the computation graph on which the evaluation is executed.</param>
        /// <param name="inputs"> The map represents input variables and their values.</param>
        /// <param name="outputs"> The map defines output variables. On return, the results are stored in Values of the map.</param>
        /// <param name="computeDevice">T he device on which the computation is executed.</param>
        /// <returns> The task representing the asynchronous operation for the evaluation.</returns>
        public static Task EvaluateAsync(this Function function, IDictionary<Variable, Value> inputs, IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            return Task.Run(() => function.Evaluate(inputs, outputs, computeDevice));
        }
    }

    public class CNTKModelPrediction
    {
        private static readonly DeviceDescriptor CPUDeviceDescriptor = DeviceDescriptor.CPUDevice;
        private readonly CNTKModelPredictionSettings modelSettings;

        public CNTKModelPrediction(string rootFolder)
        {
            modelSettings = new CNTKModelPredictionSettings()
            {
                BinFolder = rootFolder, // Path.Combine(rootFolder, "bin"),
                ModelFilename = @"Resources\models\model_cntk.pb",
                LabelsFilename = @"Resources\models\labels.txt",
                Threshold = 0.9f
            };
        }

        /// <summary>
        /// Classifiy Image using Deep Neural Networks
        /// </summary>
        /// <param name="image">image (jpeg) file to be analyzed</param>
        /// <returns>labels related to the image</returns>
        public IEnumerable<LabelConfidence> ClassifyImage(byte[] image)
        {
            // TODO: new Task
            return Process(image, modelSettings);
        }

        private IEnumerable<LabelConfidence> Process(byte[] image, CNTKModelPredictionSettings settings)
        {
            var labels = LoadLabels(settings.BinFolder, settings.LabelsFilename);
            var model = LoadModel(settings.BinFolder, settings.ModelFilename);

            var input = model.Arguments.Single();
            var output = model.Output;

            var imageTensor = LoadImageTensor(image, input);
            return Eval(model, imageTensor, input, output, labels)
                .Where(c => c.Probability >= settings.Threshold)
                .OrderByDescending(c => c.Probability);
        }

        private Tensor<float> LoadImageTensor(byte[] image, Variable input)
        {
            using (var imageStream = new MemoryStream(image))
            {
                return ConvertImageToTensorData((Bitmap)Image.FromStream(imageStream), input);
            }
        }

        private Tensor<float> ConvertImageToTensorData(Bitmap image, Variable input)
        {
            int channels = input.Shape.Dimensions[0];
            int width = input.Shape.Dimensions[1];
            int height = input.Shape.Dimensions[2];

            image = ResizeImage(image, new Size(width, height));

            Tensor<float> imageData = new DenseTensor<float>(new[] { width, height, channels }, false); // false: row-major; true: column-major; CNTK uses ColumnMajor layout

            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    Color color = image.GetPixel(x, y);
                    imageData[x, y, 0] = NormalizeRGB(color.R);
                    imageData[x, y, 1] = NormalizeRGB(color.G);
                    imageData[x, y, 2] = NormalizeRGB(color.B);
                }
            }

            return imageData;
        }

        private float NormalizeRGB(float value)
        {
            const float maxValue = 255f;
            const float mean = 2f / maxValue;
            return mean * (value - maxValue) + 1f;
        }

        private Bitmap ResizeImage(Bitmap imgToResize, Size size)
        {
            Bitmap b = new Bitmap(size.Width, size.Height);
            using (Graphics g = Graphics.FromImage(b))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(imgToResize, 0, 0, size.Width, size.Height);
            }
            return b;
        }

        private Function LoadModel(string modelsFolder, string modelFilename)
        {
            modelFilename = Path.Combine(modelsFolder, modelFilename);
            if (!File.Exists(modelFilename))
                throw new FileNotFoundException("Model file not found", modelFilename);

            Function model = null;
            try
            {
                model = Function.Load(modelFilename, CPUDeviceDescriptor);
            } catch (Exception ex)
            {
                throw ex;
            }

            return model;
        }

        private string [] LoadLabels(string modelsFolder, string labelsFilename)
        {
            labelsFilename = Path.Combine(modelsFolder, labelsFilename);
            if (!File.Exists(labelsFilename))
                throw new FileNotFoundException("Labels file not found", labelsFilename);

            var labels = File.ReadAllLines(labelsFilename);

            return labels.Where(c => !String.IsNullOrWhiteSpace(c)).ToArray();
        }

        public static IEnumerable<LabelConfidence> Eval(Function modelFunction, Tensor<float> imageData, Variable input, Variable output, string[] labels)
        {
            try
            {
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
                var probabilities = outputVal.GetDenseData<float>(output).Single().ToArray();

                var idx = 0;
                return (from label in labels
                       select new LabelConfidence { Label = label, Probability = probabilities[idx++] }).ToArray();
            }
            catch //(Exception ex)
            {
                //Debug.WriteLine(ex.ToString());
                return Enumerable.Empty<LabelConfidence>();
            }
        }

    }
}
