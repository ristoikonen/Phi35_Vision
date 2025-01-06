using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
    }
}
