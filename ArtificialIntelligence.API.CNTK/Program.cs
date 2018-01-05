using ArtificialIntelligence.API.CNTK.Services.ComputerVision.Model;
using ArtificialIntelligence.API.Services;
using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSTrainingExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine($"======== Loading model_cntk ========");
            var classifier = new CNTKModelPrediction("");
            var results = classifier.ClassifyImage(File.ReadAllBytes(@"Resources\models\parasol.jpg"));
            var label = results.Where(c => c.Probability == results.Max(d => d.Probability)).First().Label;

            Console.WriteLine($"Prediction: {label}");
        }
    }
}
