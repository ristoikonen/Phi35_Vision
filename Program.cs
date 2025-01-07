using System;
using System.Reflection.Emit;
using System.Reflection;
using System.ComponentModel;
using Microsoft.ML.OnnxRuntimeGenAI;
//using Microsoft.Extensions.Configuration.CommandLine;
using System.CommandLine;
using System.CommandLine.Invocation;
using static System.Net.Mime.MediaTypeNames;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using System.Text;


namespace Phi3
{

    class Phil3
    {

        // CMD: phi3 "C:\tmp\models\phi-3-directml-int4-awq-block-128" chat

        public const string DEFAULT_MODEL_PATH = @"C:\tmp\models\phi-3-directml-int4-awq-block-128";

        static async Task<int> Main(string[] args)
        {
            //return await ProcessImages();

            if (args.Length < 1)
                return await RunApp("", "chat");
            
            if (args.Length == 1)
                return await RunApp(args[0], "chat");
            else
                return await RunApp(args[0], args[1]);

            /*
             
            while (i < args.Length)
            {
                var arg = args[i];
                if (arg == "--non-interactive")
                {
                    interactive = false;
                }
                else if (arg == "-m")
                {
                    if (i + 1 < args.Length)
                    {
                        modelPath = Path.Combine(args[i+1]);
                    }
                }
                else if (arg == "-e")
                {
                    if (i + 1 < args.Length)
                    {
                        executionProvider = Path.Combine(args[i+1]);
                    }
                }
                i++;
            }
              
              */

        }






        static async Task<int> RunApp(string path, string? usage)
        {
            var APP_MODELUSE = Phi3.MODELUSE.IMAGE;
            if (usage is not null && usage.ToLower().StartsWith("image") || usage is not null && usage.ToLower().StartsWith("img"))
                APP_MODELUSE = MODELUSE.IMAGE;

            var general_msg = "Phi-3 Mini open model is made by Microsoft. It's 3.8 billion parameter model that's trained on synthetic data and filtered publicly available websites. It's instruction-tuned, meaning it's trained to follow different types of instructions. Due to it's size it is weak on Factual Knowledge and does not shine with languages. Then agin it's Reasoning and math is on par with GPT 3.5 Turbo models";

            var fileLocationsForImageApp = new FileLocations
            {
                ModelDirectory = @"C:\tmp\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4",  //Phi-3.5-vision-instruct
                ModelName = "Phi-3-vision-128k-instruct-onnx-cpu",
                Image1_Dir = @"C:\tmp\images\cd.jpg",
                Image2_Dir = @"C:\tmp\images\BeanStrip3.png",
                ModelUsage = "Uses CPU. Does Optical Character Recognition (OCR), Image Captioning, Table Parsing, and Reading Comprehension on Scanned Documents. Limited by its size store too much factual knowledge. Balances latency vs. accuracy. Model with acc-level-4 has better performance with a minor trade-off in accuracy."
                // https://techcommunity.microsoft.com/blog/azure-ai-services-blog/phi-3-vision-%E2%80%93-catalyzing-multimodal-innovation/4170251
            };

            var fileLocationsForChatApp = new FileLocations
            {
                ModelDirectory = @"C:\tmp\models\phi-3-directml-int4-awq-block-128",
                ModelName = "phi-3-directml-int4-awq-block-128",
                ModelUsage = "Uses CPU. Lightweight, focus on very high-quality, reasoning dense data, open model built upon datasets used for Phi-2 - synthetic data and filtered websites - with a focus on very high-quality, reasoning dense data. "
            };

            if (APP_MODELUSE == MODELUSE.CHAT)
            {
                // folder location of the ONNX model file
                // phi-3-directml-int4-awq-block-128
                // cpu-int4-rtn-block-32-acc-level-4
                var modelPath = @"C:\tmp\models\phi-3-directml-int4-awq-block-128";
                var model = new Model(modelPath);
                var tokenizer = new Tokenizer(model);

                Console.WriteLine(fileLocationsForChatApp.ToString());

                var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";

                // chat start
                Console.WriteLine(@"Ask your question. Type an empty string to Exit.");

                // chat loop
                while (true)
                {
                    // Get user question
                    Console.WriteLine();
                    Console.Write(@"Q: ");
                    var userQ = Console.ReadLine();
                    if (string.IsNullOrEmpty(userQ))
                    {
                        break;
                    }

                    // show phi3 response
                    Console.Write("Phi3: ");
                    //var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
                    var fullPrompt = $"<|system|>chemistry{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";

                    var tokens = tokenizer.Encode(fullPrompt);

                    var generatorParams = new GeneratorParams(model);
                    generatorParams.SetSearchOption("max_length", 2048);
                    generatorParams.SetSearchOption("past_present_share_buffer", false);
                    generatorParams.SetInputSequences(tokens);

                    var generator = new Generator(model, generatorParams);
                    while (!generator.IsDone())
                    {
                        generator.ComputeLogits();
                        generator.GenerateNextToken();
                        //        var e    = generator.SetActiveAdapter(GetOutput("e");
                        var outputTokens = generator.GetSequence(0);
                        var newToken = outputTokens.Slice(outputTokens.Length - 1, 1);
                        var output = tokenizer.Decode(newToken);
                        Console.Write(output);
                    }
                    Console.WriteLine();
                }
            }

            if (APP_MODELUSE == MODELUSE.IMAGE)
            {
                // path for model and images
                var modelPath = fileLocationsForImageApp.ModelDirectory;
                //@"c:\tmp\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";

                var foggyDayImagePath = fileLocationsForImageApp.Image1_Dir; // Path.Combine(Directory.GetCurrentDirectory(), "imgs", "foggyday.png");
                var petsMusicImagePath = fileLocationsForImageApp.Image2_Dir; // Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");
                var onepath = new string[] { foggyDayImagePath };
                var bothpaths = new string[] { foggyDayImagePath, petsMusicImagePath };
                var img = Images.Load(foggyDayImagePath);

                // define prompts
                var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
                string userPrompt = "Describe the image, and return the string 'STOP' at the end.";
                var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt}<|end|><|assistant|>";

                // load model and create processor
                using Model model = new Model(modelPath);
                using MultiModalProcessor processor = new MultiModalProcessor(model);
                using var tokenizerStream = processor.CreateStream();

                var tokenizer = new Tokenizer(model);

                //------


            //    StringBuilder phiResponse = new StringBuilder();


            //    var img1 = Images.Load(onepath);
            //    string userPrompt1 = "Describe the image, and return the string 'STOP' at the end.";
            //    var fullPrompt1 = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt1}<|end|><|assistant|>";

            //    // create the input tensor with the prompt and image
            //    //"visual_features": "visual_features"


            //    var inputTensors1 = processor.ProcessImages(fullPrompt1, img1);
            //    using GeneratorParams generatorParams1 = new GeneratorParams(model);
            //    generatorParams1.SetSearchOption("max_length", 3072);
            //    generatorParams1.SetInputs(inputTensors1);

            //    var isProcessingTokenStarted = false;

            //    // generate response        
            //    using var generator1 = new Generator(model, generatorParams1);
            //    while (!generator1.IsDone())
            //    {
            //        generator1.ComputeLogits();
            //        generator1.GenerateNextToken();

            //        if (!isProcessingTokenStarted)
            //        {

            //            isProcessingTokenStarted = true;
            //        }

            //        var seq = generator1.GetSequence(0)[^1];
            //        var tokenString = tokenizerStream.Decode(seq);
            //        phiResponse.Append(tokenString);
            //    }


            //Console.WriteLine(phiResponse.ToString());

            //---


            // create the input tensor with the prompt and image
            Console.WriteLine("Full Prompt: " + fullPrompt);
                Console.WriteLine("Start processing image and prompt ...");
                var inputTensors = processor.ProcessImages(fullPrompt, img);
                using GeneratorParams generatorParams = new GeneratorParams(model);
                generatorParams.SetSearchOption("max_length", 3072);
                generatorParams.SetInputs(inputTensors);

                // generate response
                Console.WriteLine("Generating response ...");
                using var generator = new Generator(model, generatorParams);
                while (!generator.IsDone())
                {
                    generator.ComputeLogits();
                    generator.GenerateNextToken();
                    var seq = generator.GetSequence(0)[^1];
                    Console.Write(tokenizerStream.Decode(seq));
                }

                Console.WriteLine("");
                Console.WriteLine("Done!");

            }
            return 0;
        }


    }
}





/*
  
 using Microsoft.ML.OnnxRuntimeGenAI;

// path for model and images
var modelPath = @"d:\phi3\models\Phi-3-vision-128k-instruct-onnx-cpu\cpu-int4-rtn-block-32-acc-level-4";

var foggyDayImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "foggyday.png");
var petsMusicImagePath = Path.Combine(Directory.GetCurrentDirectory(), "imgs", "petsmusic.png");
var img = Images.Load(petsMusicImagePath);

// define prompts
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";
string userPrompt = "Describe the image, and return the string 'STOP' at the end.";
var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|><|image_1|>{userPrompt}<|end|><|assistant|>";

// load model and create processor
using Model model = new Model(modelPath);
using MultiModalProcessor processor = new MultiModalProcessor(model);
using var tokenizerStream = processor.CreateStream();

// create the input tensor with the prompt and image
Console.WriteLine("Full Prompt: " + fullPrompt);
Console.WriteLine("Start processing image and prompt ...");
var inputTensors = processor.ProcessImages(fullPrompt, img);
using GeneratorParams generatorParams = new GeneratorParams(model);
generatorParams.SetSearchOption("max_length", 3072);
generatorParams.SetInputs(inputTensors);

// generate response
Console.WriteLine("Generating response ...");
using var generator = new Generator(model, generatorParams);
while (!generator.IsDone())
{
    generator.ComputeLogits();
    generator.GenerateNextToken();
    var seq = generator.GetSequence(0)[^1];
    Console.Write(tokenizerStream.Decode(seq));
}

Console.WriteLine("");
Console.WriteLine("Done!");
 */


/*
        var filelocationsOption = new Option<string>(
            name: "locations",
            description: "File Locations.",
            isDefault: true,
            parseArgument: result =>
            {
                if (result.Tokens.Count == 0)
                {
                    return DEFAULT_MODEL_PATH;

                }
                string? filePath = result.Tokens.Single().Value;
                if (!File.Exists(DEFAULT_MODEL_PATH))
                {
                    result.ErrorMessage = "DEFAULT_MODEL_PATH file does not exist";
                    return "";
                }
                else
                {
                    return DEFAULT_MODEL_PATH;
                }
            });

        var usageOption = new Option<string>(
            name: "usage",
            description: "Model usage; chat or image.",
            getDefaultValue: () => "chat");

        var rootCommand = new RootCommand("Phi3 AI App");
        
        var locationsCommand = new Command("path", "Path to model files.")
                            {
                            filelocationsOption
                            };

        locationsCommand.SetHandler(async (locations) =>
        {
            await RunApp(locations, null);
        },
        filelocationsOption);

        rootCommand.AddCommand(locationsCommand);


      
        
        //var usage_Command = new Command("usage", "Usage: write 'chat' or 'image'.")
        //                    {
        //                    usageOption
        //                    };
        //locationsCommand.AddCommand(usage_Command);
        



        locationsCommand.SetHandler(async (file, usage) =>
        {
            await RunApp(file!, usage);
        },
        filelocationsOption, usageOption);
        
        

        return await rootCommand.InvokeAsync(args);
        */