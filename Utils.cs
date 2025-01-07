using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Phi3
{


    public enum MODELUSE : ushort
    {
        CHAT,
        IMAGE
    }

    

    public interface IFileLocations
    {
        public string ModelDirectory
        { get; set; }

        public string ModelName
        { get; set; }


    }

    public record FileLocations //: IFileLocations
    {
        private readonly string rootdirectory;

        // public required string ModelDirectory
        public FileLocations()
        {

        }
        public FileLocations(string ModelDirectory)
        {
            this.ModelDirectory = ModelDirectory;
        }
        public FileLocations(string ModelDirectory, string ModelName)
        {
            this.ModelDirectory = ModelDirectory;
            this.ModelName = ModelName;
        }
        public string ModelDirectory
        {
            get; set;
            //get => rootdirectory;
            //init => rootdirectory = @"c:\tmp"
            //set;
        }

        public string ModelName
        { get; set; }
        public string ModelUsage
        { get; set; }

        public string Image1_Dir
        { get; set; }

        public string Image2_Dir
        { get; set; }

        public override string ToString()
        {
            var str = "ModelDirectory = " + (this.ModelDirectory ?? "");
            str += Environment.NewLine + "ModelName = " + (this.ModelName ?? "");
            str += Environment.NewLine + "ModelUsage = " + (this.ModelUsage ?? "");
            return str;
        }
    


        int ProcessImages()
        {

            //string modelPath = "path_to_your_model.onnx";
            //string[] imagePaths = { "image1.jpg", "image2.jpg" }; // Add your image paths here
            string textInput = "Your text input here";
            var fileLocationsForImageApp = new FileLocations
            {
                ModelDirectory = @"C:\tmp\models\Phi-3.5-vision-instruct", //Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4",
                ModelName = "Phi-3.5-vision-instruct",
                Image1_Dir = @"C:\tmp\images\BayRoad.png",
                Image2_Dir = @"C:\tmp\images\BeanStrip3.png",
                ModelUsage = "Uses CPU. Does Optical Character Recognition (OCR), Image Captioning, Table Parsing, and Reading Comprehension on Scanned Documents. Limited by its size store too much factual knowledge. Balances latency vs. accuracy. Model with acc-level-4 has better performance with a minor trade-off in accuracy."
                // https://techcommunity.microsoft.com/blog/azure-ai-services-blog/phi-3-vision-%E2%80%93-catalyzing-multimodal-innovation/4170251
            };

            string[] imagePaths = { fileLocationsForImageApp.Image1_Dir, fileLocationsForImageApp.Image2_Dir };

            using var session = new InferenceSession(fileLocationsForImageApp.ModelDirectory);

            foreach (var imagePath in imagePaths)
            {
                var inputTensor = LoadImageAsTensor(imagePath);
                var textTensor = LoadTextAsTensor(textInput);

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image_input", inputTensor),
                    NamedOnnxValue.CreateFromTensor("text_input", textTensor)
                };

                using var results =  session.Run(inputs);
                var output = results.First().AsTensor<float>().ToArray();
                Console.WriteLine($"Processed {imagePath}: {string.Join(", ", output)}");
            }
            return 0;
        }

        DenseTensor<float> LoadImageAsTensor(string imagePath)
        {
            using var bitmap = new Bitmap(imagePath);
            var tensor = new DenseTensor<float>(new[] { 1, 3, bitmap.Height, bitmap.Width });

            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    var color = bitmap.GetPixel(x, y);
                    tensor[0, 0, y, x] = color.R / 255.0f;
                    tensor[0, 1, y, x] = color.G / 255.0f;
                    tensor[0, 2, y, x] = color.B / 255.0f;
                }
            }

            return tensor;
        }

        DenseTensor<float> LoadTextAsTensor(string text)
        {
            var words = text.Split(' ');
            var tensor = new DenseTensor<float>(new[] { 1, words.Length });

            for (int i = 0; i < words.Length; i++)
            {
                tensor[0, i] = ConvertWordToFloat(words[i]);
            }

            return tensor;
        }



        float ConvertWordToFloat(string word)
        {
            // Simple example: convert each character to its ASCII value and sum them up
            float sum = 0;
            foreach (var ch in word)
            {
                sum += ch;
            }
            return sum;
        }
    }
}
