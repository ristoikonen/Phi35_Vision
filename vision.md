# Vision Language Model



```c#

using var session = new InferenceSession(modelPath);

foreach (var imagePath in imagePaths)
{
    var inputTensor = LoadImageAsTensor(imagePath);
    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input", inputTensor)
    };

    using var results = session.Run(inputs);
    var output = results.First().AsTensor<float>().ToArray();
    Console.WriteLine($"Processed {imagePath}: {string.Join(", ", output)}");
}
...

// LoadImageAsTensor
// Load percentages; how much of pure red does this pixel have, into RGB tensor


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

```

### Source: https://huggingface.co/blog/vlms

Vision language models are broadly defined as multimodal models that can learn from images and text. 
They are a type of generative models that take image and text inputs, and generate text outputs. 
Large vision language models have good zero-shot capabilities, generalize well, and can work with many types of images, including documents, 
web pages, and more. The use cases include chatting about images, image recognition via instructions, visual question answering, document understanding, image captioning, and others. 
Some vision language models can also capture spatial properties in an image. 
These models can output bounding boxes or segmentation masks when prompted to detect or segment a particular subject, 
or they can localize different entities or answer questions about their relative or absolute positions. 
There’s a lot of diversity within the existing set of large vision language models, 
the data they were trained on, how they encode images, and, thus, their capabilities.

## Vision training

1. Localisation
2. Segments
3. QA

=> Birds in the image are three Cocatoos, they are large white birds with yellow crest. 




## Overview of Open-source Vision Language Models
There are many open vision language models on the Hugging Face Hub. Some of the most prominent ones are shown in the table below.

There are base models, and models fine-tuned for chat that can be used in conversational mode.
Some of these models have a feature called “grounding” which reduces model hallucinations.
All models are trained on English unless stated otherwise.

Finding the right Vision Language Model
There are many ways to select the most appropriate model for your use case.

Vision Arena is a leaderboard solely based on anonymous voting of model outputs and is updated continuously. 
In this arena, the users enter an image and a prompt, and outputs from two different models are sampled anonymously, 
then the user can pick their preferred output. This way, the leaderboard is constructed solely based on human preferences.

## Technical Details
There are various ways to pretrain a vision language model. The main trick is to unify the image and text representation and feed it to a text decoder 
for generation. The most common and prominent models often consist of an image encoder, an embedding projector to align image and text representations 
(often a dense neural network) and a text decoder stacked in this order. As for the training parts, 
different models have been following different approaches.

For instance, LLaVA consists of a CLIP image encoder, a multimodal projector and a Vicuna text decoder. 
The authors fed a dataset of images and captions to GPT-4 and generated questions related to the caption and the image. 
The authors have frozen the image encoder and text decoder and have only trained the multimodal projector to align the image and text features 
by feeding the model images and generated questions and comparing the model output to the ground truth captions. After the projector pretraining, 
they keep the image encoder frozen, unfreeze the text decoder, and train the projector with the decoder. 
This way of pre-training and fine-tuning is the most common way of training vision language models.


