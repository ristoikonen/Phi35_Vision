using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Spectre.Console;

namespace Phi3
{
    public static class SpectreConsoleOutput
    {
        public static void DisplayTitle(string title = "Phi-3.5 Text and Vision")
        {
            AnsiConsole.Write(new FigletText(title).Centered().Color(Color.Purple));
        }

        public static void DisplayWait(string title = "Loading the Phi-3.5 model")
        {
            var panel = new Panel(title)
            {
                Border = BoxBorder.Rounded,
                Padding = new Padding(10, 1, 1, 1),
                BorderStyle = new Style(Color.Red)
                
            };

            AnsiConsole.Write(panel);
        }

        public static void ClearDisplay()
        {
            AnsiConsole.Clear();
        }

        public static void DisplayTitleH2(string subtitle)
        {
            AnsiConsole.MarkupLine($"[bold][blue]=== {subtitle} ===[/][/]");
            AnsiConsole.MarkupLine($"");
        }

        public static void DisplayTitleH3(string subtitle)
        {
            AnsiConsole.MarkupLine($"[bold]>> {subtitle}[/]");
            AnsiConsole.MarkupLine($"");
        }

        public static void DisplayQuestion(string question)
        {
            AnsiConsole.MarkupLine($"[bold][blue]>>Q: {question}[/][/]");
            AnsiConsole.MarkupLine($"");
        }
        public static void DisplayAnswerStart(string answerPrefix)
        {
            AnsiConsole.Markup($"[bold][blue]>> {answerPrefix}:[/][/]");
        }

        public static void DisplayFilePath(string prefix, string filePath)
        {
            var path = new TextPath(filePath);

            AnsiConsole.Markup($"[bold][blue]>> {prefix}: [/][/]");
            AnsiConsole.Write(path);
            AnsiConsole.MarkupLine($"");
        }

        public static void DisplaySubtitle(string prefix, string content)
        {
            AnsiConsole.Markup($"[bold][blue]>> {prefix}: [/][/]");
            AnsiConsole.WriteLine(content);
            AnsiConsole.MarkupLine($"");
        }

        //public static int ShowProgress(string question)
        //{
        //    var number = AnsiConsole.Progress();
        //    number.Start(f => f.AddTaskBefore())
        //    return 1;
        //}

        public static int AskForNumber(string question)
        {
            var number = AnsiConsole.Ask<int>(@$"[green]{question}[/]");
            return number;
        }

        public static string AskForString(string question)
        {
            var response = AnsiConsole.Ask<string>(@$"[green]{question}[/]");
            return response;
        }

        public static List<string> SelectScenarios()
        {
            // Ask for the user's favorite fruits
            var scenarios = AnsiConsole.Prompt(
                new MultiSelectionPrompt<string>()
                    .Title("Select the [green]Phi 3.5 Vision scenarios[/] to run? Phi-3.5 Mini open model is made by Microsoft. It's 3.8 billion parameter model that's trained on synthetic data and filtered publicly available websites. It's instruction-tuned, meaning it's trained to follow different types of instructions. Due to it's size it is weak on Factual Knowledge and does not shine with languages. Then agin it's Reasoning and math is on par with GPT 3.5 Turbo models")
                    .PageSize(10)
                    .Required(true)
                    .MoreChoicesText("[grey](Move up and down to reveal more scenarios)[/]")
                    .InstructionsText(
                        "[grey](Press [blue]<space>[/] to toggle a scenario, " +
                        "[green]<enter>[/] to accept)[/]")
                    .AddChoiceGroup("Select an image to be analuyzed", new[]
                        {"BeanStrip3.png","cd.jpg","BayRoad.png",
                        })
                    .AddChoices(new[] {
                    "Type the image path to be analyzed",
                    "Type a question"
                        })
                    );
            return scenarios;
        }
    }
}
