using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;


using NumSharp;

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

    public record FileLocations
    {
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
            get; 
            //init => ModelDirectory = @"c:\tmp";
            set;
        }

        public string ModelName
        { get; set; }
        public string ModelDesc
        { get; set; }

        public string Image1_Dir
        { get; set; }

        public string Image2_Dir
        { get; set; }

        public MODELUSE ModelUsage
        { get; set; }

        public override string ToString()
        {
            var str = "ModelDirectory = " + (this.ModelDirectory ?? "");
            str += Environment.NewLine + "ModelName = " + (this.ModelName ?? "");
            str += Environment.NewLine + "ModelUsage = " + (this.ModelDesc ?? "");
            return str;
        }
    }




/*

    public class UNet
    {
        public static List<NamedOnnxValue> CreateUnetModelInput(Tensor<float> encoderHiddenStates, Tensor<float> sample, long timeStep)
        {

            var input = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderHiddenStates),
                NamedOnnxValue.CreateFromTensor("sample", sample),
                NamedOnnxValue.CreateFromTensor("timestep", new DenseTensor<long>(new long[] { timeStep }, new int[] { 1 }))
            };

            return input;

        }

        public static Tensor<float> GenerateLatentSample(StableDiffusionConfig config, int seed, float initNoiseSigma)
        {
            return GenerateLatentSample(config.Height, config.Width, seed, initNoiseSigma);
        }
        public static Tensor<float> GenerateLatentSample(int height, int width, int seed, float initNoiseSigma)
        {
            var random = new Random(seed);
            var batchSize = 1;
            var channels = 4;
            var latents = new DenseTensor<float>(new[] { batchSize, channels, height / 8, width / 8 });
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number

                // add noise to latents with * scheduler.init_noise_sigma
                // generate randoms that are negative and positive
                latentsArray[i] = (float)standardNormalRand * initNoiseSigma;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }

        private static Tensor<float> performGuidance(Tensor<float> noisePred, Tensor<float> noisePredText, double guidanceScale)
        {
            for (int i = 0; i < noisePred.Dimensions[0]; i++)
            {
                for (int j = 0; j < noisePred.Dimensions[1]; j++)
                {
                    for (int k = 0; k < noisePred.Dimensions[2]; k++)
                    {
                        for (int l = 0; l < noisePred.Dimensions[3]; l++)
                        {
                            noisePred[i, j, k, l] = noisePred[i, j, k, l] + (float)guidanceScale * (noisePredText[i, j, k, l] - noisePred[i, j, k, l]);
                        }
                    }
                }
            }
            return noisePred;
        }

        public static SixLabors.ImageSharp.Image Inference(String prompt, StableDiffusionConfig config)
        {
            // Preprocess text
            var textEmbeddings = TextProcessing.PreprocessText(prompt, config);

            var scheduler = new LMSDiscreteScheduler();
            //var scheduler = new EulerAncestralDiscreteScheduler();
            var timesteps = scheduler.SetTimesteps(config.NumInferenceSteps);
            //  If you use the same seed, you will get the same image result.
            var seed = new Random().Next();
            //var seed = 329922609;
            Console.WriteLine($"Seed generated: {seed}");
            // create latent tensor

            var latents = GenerateLatentSample(config, seed, scheduler.InitNoiseSigma);

            var sessionOptions = config.GetSessionOptionsForEp();
            // Create Inference Session
            var unetSession = new InferenceSession(config.UnetOnnxPath, sessionOptions);

            var input = new List<NamedOnnxValue>();
            for (int t = 0; t < timesteps.Length; t++)
            {
                // torch.cat([latents] * 2)
                var latentModelInput = TensorHelper.Duplicate(latents.ToArray(), new[] { 2, 4, config.Height / 8, config.Width / 8 });

                // latent_model_input = scheduler.scale_model_input(latent_model_input, timestep = t)
                latentModelInput = scheduler.ScaleInput(latentModelInput, timesteps[t]);

                Console.WriteLine($"scaled model input {latentModelInput[0]} at step {t}. Max {latentModelInput.Max()} Min{latentModelInput.Min()}");
                input = CreateUnetModelInput(textEmbeddings, latentModelInput, timesteps[t]);

                // Run Inference
                var output = unetSession.Run(input);
                var outputTensor = (output.ToList().First().Value as DenseTensor<float>);

                // Split tensors from 2,4,64,64 to 1,4,64,64
                var splitTensors = TensorHelper.SplitTensor(outputTensor, new[] { 1, 4, config.Height / 8, config.Width / 8 });
                var noisePred = splitTensors.Item1;
                var noisePredText = splitTensors.Item2;

                // Perform guidance
                noisePred = performGuidance(noisePred, noisePredText, config.GuidanceScale);

                // LMS Scheduler Step
                latents = scheduler.Step(noisePred, timesteps[t], latents);
                Console.WriteLine($"latents result after step {t} min {latents.Min()} max {latents.Max()}");

            }

            // Scale and decode the image latents with vae.
            // latents = 1 / 0.18215 * latents
            latents = TensorHelper.MultipleTensorByFloat(latents.ToArray(), (1.0f / 0.18215f), latents.Dimensions.ToArray());
            var decoderInput = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("latent_sample", latents) };

            // Decode image
            var imageResultTensor = VaeDecoder.Decoder(decoderInput, config);
            var isNotSafe = SafetyChecker.IsNotSafe(imageResultTensor, config);

            if (isNotSafe)
            {
                return null;

            }
            var image = VaeDecoder.ConvertToImage(imageResultTensor, config);
            return image;

        }

    }





    public class StableDiffusionConfig
    {

        public enum ExecutionProvider
        {
            DirectML = 0,
            Cuda = 1,
            Cpu = 2
        }
        // default props
        public int NumInferenceSteps = 15;
        public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.Cuda;
        public double GuidanceScale = 7.5;
        public int Height = 512;
        public int Width = 512;
        public int DeviceId = 0;


        public string TokenizerOnnxPath = "cliptokenizer.onnx";
        public string TextEncoderOnnxPath = "";
        public string UnetOnnxPath = "";
        public string VaeDecoderOnnxPath = "";
        public string SafetyModelPath = "";

        // default directory for images
        public string ImageOutputPath = "";

        public SessionOptions GetSessionOptionsForEp()
        {
            var sessionOptions = new SessionOptions();


            switch (this.ExecutionProviderTarget)
            {
                case ExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    //default to CUDA, fall back on CPU if CUDA is not available.
                    sessionOptions.AppendExecutionProvider_CUDA(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                    return sessionOptions;

            }

        }



    }


    public class TensorHelper
    {
        public static DenseTensor<T> CreateTensor<T>(T[] data, int[] dimensions)
        {
            return new DenseTensor<T>(data, dimensions); ;
        }

        public static DenseTensor<float> DivideTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] / value;
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(float[] data, float value, int[] dimensions)
        {
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = data[i] * value;
            }

            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> MultipleTensorByFloat(Tensor<float> data, float value)
        {
            return MultipleTensorByFloat(data.ToArray(), value, data.Dimensions.ToArray());
        }

        public static DenseTensor<float> AddTensors(float[] sample, float[] sumTensor, int[] dimensions)
        {
            for (var i = 0; i < sample.Length; i++)
            {
                sample[i] = sample[i] + sumTensor[i];
            }
            return CreateTensor(sample, dimensions); ;
        }

        public static DenseTensor<float> AddTensors(Tensor<float> sample, Tensor<float> sumTensor)
        {
            return AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static Tuple<Tensor<float>, Tensor<float>> SplitTensor(Tensor<float> tensorToSplit, int[] dimensions)
        {
            var tensor1 = new DenseTensor<float>(dimensions);
            var tensor2 = new DenseTensor<float>(dimensions);

            for (int i = 0; i < 1; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    for (int k = 0; k < 512 / 8; k++)
                    {
                        for (int l = 0; l < 512 / 8; l++)
                        {
                            tensor1[i, j, k, l] = tensorToSplit[i, j, k, l];
                            tensor2[i, j, k, l] = tensorToSplit[i, j + 4, k, l];
                        }
                    }
                }
            }
            return new Tuple<Tensor<float>, Tensor<float>>(tensor1, tensor2);

        }

        public static DenseTensor<float> SumTensors(Tensor<float>[] tensorArray, int[] dimensions)
        {
            var sumTensor = new DenseTensor<float>(dimensions);
            var sumArray = new float[sumTensor.Length];

            for (int m = 0; m < tensorArray.Count(); m++)
            {
                var tensorToSum = tensorArray[m].ToArray();
                for (var i = 0; i < tensorToSum.Length; i++)
                {
                    sumArray[i] += (float)tensorToSum[i];
                }
            }

            return CreateTensor(sumArray, dimensions);
        }

        public static DenseTensor<float> Duplicate(float[] data, int[] dimensions)
        {
            data = data.Concat(data).ToArray();
            return CreateTensor(data, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(float[] sample, float[] subTensor, int[] dimensions)
        {
            for (var i = 0; i < sample.Length; i++)
            {
                sample[i] = sample[i] - subTensor[i];
            }
            return CreateTensor(sample, dimensions);
        }

        public static DenseTensor<float> SubtractTensors(Tensor<float> sample, Tensor<float> subTensor)
        {
            return SubtractTensors(sample.ToArray(), subTensor.ToArray(), sample.Dimensions.ToArray());
        }

        public static Tensor<float> GetRandomTensor(ReadOnlySpan<int> dimensions)
        {
            var random = new Random();
            var latents = new DenseTensor<float>(dimensions);
            var latentsArray = latents.ToArray();

            for (int i = 0; i < latentsArray.Length; i++)
            {
                // Generate a random number from a normal distribution with mean 0 and variance 1
                var u1 = random.NextDouble(); // Uniform(0,1) random number
                var u2 = random.NextDouble(); // Uniform(0,1) random number
                var radius = Math.Sqrt(-2.0 * Math.Log(u1)); // Radius of polar coordinates
                var theta = 2.0 * Math.PI * u2; // Angle of polar coordinates
                var standardNormalRand = radius * Math.Cos(theta); // Standard normal random number
                latentsArray[i] = (float)standardNormalRand;
            }

            latents = TensorHelper.CreateTensor(latentsArray, latents.Dimensions.ToArray());

            return latents;

        }
    }



    public static class TextProcessing
    {
        public static DenseTensor<float> PreprocessText(String prompt, StableDiffusionConfig config)
        {
            // Load the tokenizer and text encoder to tokenize and encode the text.
            var textTokenized = TokenizeText(prompt, config);
            var textPromptEmbeddings = TextEncoder(textTokenized, config).ToArray();

            // Create uncond_input of blank tokens
            var uncondInputTokens = CreateUncondInput();
            var uncondEmbedding = TextEncoder(uncondInputTokens, config).ToArray();

            // Concant textEmeddings and uncondEmbedding
            DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

            for (var i = 0; i < textPromptEmbeddings.Length; i++)
            {
                textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
                textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
            }
            return textEmbeddings;
        }
        public static int[] TokenizeText(string text, StableDiffusionConfig config)
        {
            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();

            // Create an InferenceSession from the onnx clip tokenizer.
            var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);
            var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };
            // Run session and send the input data in to get inference output. 
            var tokens = tokenizeSession.Run(inputString);


            var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
            Console.WriteLine(String.Join(" ", inputIds));

            // Cast inputIds to Int32
            var InputIdsInt = inputIds.Select(x => (int)x).ToArray();

            var modelMaxLength = 77;
            // Pad array with 49407 until length is modelMaxLength
            if (InputIdsInt.Length < modelMaxLength)
            {
                var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
                InputIdsInt = InputIdsInt.Concat(pad).ToArray();
            }

            return InputIdsInt;

        }

        public static int[] CreateUncondInput()
        {
            // Create an array of empty tokens for the unconditional input.
            var blankTokenValue = 49407;
            var modelMaxLength = 77;
            var inputIds = new List<Int32>();
            inputIds.Add(49406);
            var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
            inputIds.AddRange(pad);

            return inputIds.ToArray();
        }

        public static DenseTensor<float> TextEncoder(int[] tokenizedInput, StableDiffusionConfig config)
        {
            // Create input tensor.
            var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });

            var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

            // Set CUDA EP
            var sessionOptions = config.GetSessionOptionsForEp();

            var encodeSession = new InferenceSession(config.TextEncoderOnnxPath, sessionOptions);
            // Run inference.
            var encoded = encodeSession.Run(input);

            var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
            var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

            return lastHiddenStateTensor;

        }

    }

    public static class VaeDecoder
    {
        public static Tensor<float> Decoder(List<NamedOnnxValue> input, StableDiffusionConfig config)
        {
            config.ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cpu;
            var sessionOptions = config.GetSessionOptionsForEp();
            // Create an InferenceSession from the Model Path.
            var vaeDecodeSession = new InferenceSession(config.VaeDecoderOnnxPath, sessionOptions);

            // Run session and send the input data in to get inference output. 
            var output = vaeDecodeSession.Run(input);
            var result = (output.ToList().First().Value as Tensor<float>);

            return result;
        }

        // create method to convert float array to an image with imagesharp
        public static Image<Rgba32> ConvertToImage(Tensor<float> output, StableDiffusionConfig config, int width = 512, int height = 512)
        {
            var result = new Image<Rgba32>(width, height);

            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    result[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((output[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((output[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            var imageName = $"sd_image_{DateTime.Now.ToString("yyyyMMddHHmm")}.png";
            var imagePath = Path.Combine(Directory.GetCurrentDirectory(), config.ImageOutputPath, imageName);

            result.Save(imagePath);

            Console.WriteLine($"Image saved to: {imagePath}");

            return result;
        }
    }


    public class LMSDiscreteScheduler : SchedulerBase
    {
        private int _numTrainTimesteps;
        private string _predictionType;

        public override Tensor<float> Sigmas { get; set; }
        public override List<int> Timesteps { get; set; }
        public List<Tensor<float>> Derivatives;
        public override float InitNoiseSigma { get; set; }

        public LMSDiscreteScheduler(int num_train_timesteps = 1000, float beta_start = 0.00085f, float beta_end = 0.012f, string beta_schedule = "scaled_linear", string prediction_type = "epsilon", List<float> trained_betas = null)
        {
            _numTrainTimesteps = num_train_timesteps;
            _predictionType = prediction_type;
            Derivatives = new List<Tensor<float>>();
            Timesteps = new List<int>();

            var alphas = new List<float>();
            var betas = new List<float>();

            if (trained_betas != null)
            {
                betas = trained_betas;
            }
            else if (beta_schedule == "linear")
            {
                betas = Enumerable.Range(0, num_train_timesteps).Select(i => beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1)).ToList();
            }
            else if (beta_schedule == "scaled_linear")
            {
                var start = (float)Math.Sqrt(beta_start);
                var end = (float)Math.Sqrt(beta_end);
                betas = np.linspace(start, end, num_train_timesteps).ToArray<float>().Select(x => x * x).ToList();

            }
            else
            {
                throw new Exception("beta_schedule must be one of 'linear' or 'scaled_linear'");
            }

            alphas = betas.Select(beta => 1 - beta).ToList();

            this._alphasCumulativeProducts = alphas.Select((alpha, i) => alphas.Take(i + 1).Aggregate((a, b) => a * b)).ToList();
            // Create sigmas as a list and reverse it
            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();

            // standard deviation of the initial noise distrubution
            this.InitNoiseSigma = (float)sigmas.Max();

        }

        //python line 135 of scheduling_lms_discrete.py
        public double GetLmsCoefficient(int order, int t, int currentOrder)
        {
            // Compute a linear multistep coefficient.

            double LmsDerivative(double tau)
            {
                double prod = 1.0;
                for (int k = 0; k < order; k++)
                {
                    if (currentOrder == k)
                    {
                        continue;
                    }
                    prod *= (tau - this.Sigmas[t - k]) / (this.Sigmas[t - currentOrder] - this.Sigmas[t - k]);
                }
                return prod;
            }

            double integratedCoeff = Integrate.OnClosedInterval(LmsDerivative, this.Sigmas[t], this.Sigmas[t + 1], 1e-4);

            return integratedCoeff;
        }

        // Line 157 of scheduling_lms_discrete.py from HuggingFace diffusers
        public override int[] SetTimesteps(int num_inference_steps)
        {
            double start = 0;
            double stop = _numTrainTimesteps - 1;
            double[] timesteps = np.linspace(start, stop, num_inference_steps).ToArray<double>();

            this.Timesteps = timesteps.Select(x => (int)x).Reverse().ToList();

            var sigmas = _alphasCumulativeProducts.Select(alpha_prod => Math.Sqrt((1 - alpha_prod) / alpha_prod)).Reverse().ToList();
            var range = np.arange((double)0, (double)(sigmas.Count)).ToArray<double>();
            sigmas = Interpolate(timesteps, range, sigmas).ToList();
            this.Sigmas = new DenseTensor<float>(sigmas.Count());
            for (int i = 0; i < sigmas.Count(); i++)
            {
                this.Sigmas[i] = (float)sigmas[i];
            }
            return this.Timesteps.ToArray();

        }

        public override DenseTensor<float> Step(
               Tensor<float> modelOutput,
               int timestep,
               Tensor<float> sample,
               int order = 4)
        {
            int stepIndex = this.Timesteps.IndexOf(timestep);
            var sigma = this.Sigmas[stepIndex];

            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            Tensor<float> predOriginalSample;

            // Create array of type float length modelOutput.length
            float[] predOriginalSampleArray = new float[modelOutput.Length];
            var modelOutPutArray = modelOutput.ToArray();
            var sampleArray = sample.ToArray();

            if (this._predictionType == "epsilon")
            {

                for (int i = 0; i < modelOutPutArray.Length; i++)
                {
                    predOriginalSampleArray[i] = sampleArray[i] - sigma * modelOutPutArray[i];
                }
                predOriginalSample = TensorHelper.CreateTensor(predOriginalSampleArray, modelOutput.Dimensions.ToArray());

            }
            else if (this._predictionType == "v_prediction")
            {
                //predOriginalSample = modelOutput * ((-sigma / Math.Sqrt((Math.Pow(sigma,2) + 1))) + (sample / (Math.Pow(sigma,2) + 1)));
                throw new Exception($"prediction_type given as {this._predictionType} not implemented yet.");
            }
            else
            {
                throw new Exception($"prediction_type given as {this._predictionType} must be one of `epsilon`, or `v_prediction`");
            }

            // 2. Convert to an ODE derivative
            var derivativeItems = new DenseTensor<float>(sample.Dimensions.ToArray());

            var derivativeItemsArray = new float[derivativeItems.Length];

            for (int i = 0; i < modelOutPutArray.Length; i++)
            {
                //predOriginalSample = (sample - predOriginalSample) / sigma;
                derivativeItemsArray[i] = (sampleArray[i] - predOriginalSampleArray[i]) / sigma;
            }
            derivativeItems = TensorHelper.CreateTensor(derivativeItemsArray, derivativeItems.Dimensions.ToArray());

            this.Derivatives?.Add(derivativeItems);

            if (this.Derivatives?.Count() > order)
            {
                // remove first element
                this.Derivatives?.RemoveAt(0);
            }

            // 3. compute linear multistep coefficients
            order = Math.Min(stepIndex + 1, order);
            var lmsCoeffs = Enumerable.Range(0, order).Select(currOrder => GetLmsCoefficient(order, stepIndex, currOrder)).ToArray();

            // 4. compute previous sample based on the derivative path
            // Reverse list of tensors this.derivatives
            var revDerivatives = Enumerable.Reverse(this.Derivatives).ToList();

            // Create list of tuples from the lmsCoeffs and reversed derivatives
            var lmsCoeffsAndDerivatives = lmsCoeffs.Zip(revDerivatives, (lmsCoeff, derivative) => (lmsCoeff, derivative));

            // Create tensor for product of lmscoeffs and derivatives
            var lmsDerProduct = new Tensor<float>[this.Derivatives.Count()];

            for (int m = 0; m < lmsCoeffsAndDerivatives.Count(); m++)
            {
                var item = lmsCoeffsAndDerivatives.ElementAt(m);
                // Multiply to coeff by each derivatives to create the new tensors
                lmsDerProduct[m] = TensorHelper.MultipleTensorByFloat(item.derivative.ToArray(), (float)item.lmsCoeff, item.derivative.Dimensions.ToArray());
            }
            // Sum the tensors
            var sumTensor = TensorHelper.SumTensors(lmsDerProduct, new[] { 1, 4, 64, 64 });

            // Add the sumed tensor to the sample
            var prevSample = TensorHelper.AddTensors(sample.ToArray(), sumTensor.ToArray(), sample.Dimensions.ToArray());

            Console.WriteLine(prevSample[0]);
            return prevSample;

        }
    }


    public static class SafetyChecker
    {
        public static bool IsNotSafe(Tensor<float> resultImage, StableDiffusionConfig config)
        {
            //clip input
            var inputTensor = ClipImageFeatureExtractor(resultImage, config);
            //images input
            var inputImagesTensor = ReorderTensor(inputTensor);

            var input = new List<NamedOnnxValue> { //batch channel height width
                                                    NamedOnnxValue.CreateFromTensor("clip_input", inputTensor),
                                                    //batch, height, width, channel
                                                    NamedOnnxValue.CreateFromTensor("images", inputImagesTensor)};

            var sessionOptions = config.GetSessionOptionsForEp();
            var session = new InferenceSession(config.SafetyModelPath, sessionOptions);

            // Run session and send the input data in to get inference output. 
            var output = session.Run(input);
            var result = (output.ToList().Last().Value as IEnumerable<bool>).ToArray()[0];

            return result;
        }

        private static DenseTensor<float> ReorderTensor(Tensor<float> inputTensor)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }

            return inputImagesTensor;
        }
        private static DenseTensor<float> ClipImageFeatureExtractor(Tensor<float> imageTensor, StableDiffusionConfig config)
        {
            // Read image
            //using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            //convert tensor result to image
            var image = new SixLabors.ImageSharp.Image<Rgba32>(config.Width, config.Height);

            for (var y = 0; y < config.Height; y++)
            {
                for (var x = 0; x < config.Width; x++)
                {
                    image[x, y] = new Rgba32(
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                        (byte)(Math.Round(Math.Clamp((imageTensor[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                    );
                }
            }

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess image
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };
            for (int y = 0; y < image.Height; y++)
            {
                Span<Rgba32> pixelSpan = image.GetPixelRowSpan(y);

                for (int x = 0; x < image.Width; x++)
                {
                    input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                    input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                    input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                }
            }

            return input;

        }
    }




    public abstract class SchedulerBase
    {
        protected readonly int _numTrainTimesteps;
        protected List<float> _alphasCumulativeProducts;
        public bool is_scale_input_called;

        public abstract List<int> Timesteps { get; set; }
        public abstract Tensor<float> Sigmas { get; set; }
        public abstract float InitNoiseSigma { get; set; }

        public SchedulerBase(int _numTrainTimesteps = 1000)
        {
            this._numTrainTimesteps = _numTrainTimesteps;
        }

        public static double[] Interpolate(double[] timesteps, double[] range, List<double> sigmas)
        {

            // Create an output array with the same shape as timesteps
            var result = np.zeros(timesteps.Length + 1);

            // Loop over each element of timesteps
            for (int i = 0; i < timesteps.Length; i++)
            {
                // Find the index of the first element in range that is greater than or equal to timesteps[i]
                int index = Array.BinarySearch(range, timesteps[i]);

                // If timesteps[i] is exactly equal to an element in range, use the corresponding value in sigma
                if (index >= 0)
                {
                    result[i] = sigmas[index];
                }

                // If timesteps[i] is less than the first element in range, use the first value in sigmas
                else if (index == -1)
                {
                    result[i] = sigmas[0];
                }

                // If timesteps[i] is greater than the last element in range, use the last value in sigmas
                else if (index == -range.Length - 1)
                {
                    result[i] = sigmas[-1];
                }

                // Otherwise, interpolate linearly between two adjacent values in sigmas
                else
                {
                    index = ~index; // bitwise complement of j gives the insertion point of x[i]
                    double t = (timesteps[i] - range[index - 1]) / (range[index] - range[index - 1]); // fractional distance between two points
                    result[i] = sigmas[index - 1] + t * (sigmas[index] - sigmas[index - 1]); // linear interpolation formula
                }

            }
            //  add 0.000 to the end of the result
            result = np.add(result, 0.000f);

            return result.ToArray<double>();
        }

        public DenseTensor<float> ScaleInput(DenseTensor<float> sample, int timestep)
        {
            // Get step index of timestep from TimeSteps
            int stepIndex = this.Timesteps.IndexOf(timestep);
            // Get sigma at stepIndex
            var sigma = this.Sigmas[stepIndex];
            sigma = (float)Math.Sqrt((Math.Pow(sigma, 2) + 1));

            // Divide sample tensor shape {2,4,64,64} by sigma
            sample = TensorHelper.DivideTensorByFloat(sample.ToArray(), sigma, sample.Dimensions.ToArray());
            is_scale_input_called = true;
            return sample;
        }
        public abstract int[] SetTimesteps(int num_inference_steps);

        public abstract DenseTensor<float> Step(
               Tensor<float> modelOutput,
               int timestep,
               Tensor<float> sample,
               int order = 4);
    }


    */














}
